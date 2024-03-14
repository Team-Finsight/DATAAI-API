import os
from dotenv import load_dotenv
import uuid
from werkzeug.utils import secure_filename
from datetime import datetime
import pandas as pd
from pandasai.connectors import MySQLConnector
from pandasai.llm import OpenAI
from pandasai import SmartDataframe, SmartDatalake
from flask import Flask, request, jsonify
from pandasai.responses.response_parser import ResponseParser
from flask import make_response
from flask import send_file
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
from flask import Flask, send_file, request, Response, jsonify
import json
matplotlib.use('agg')

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)

UPLOAD_FOLDER = 'uploaded_files'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class PandasDataFrame(ResponseParser):
    def __init__(self, context=None):
        self.context = context

    def format_dataframe(self, result):
        if 'value' in result and isinstance(result['value'], pd.DataFrame):
            # Convert DataFrame to JSON
            response_json = result['value'].to_json(orient='records')
            # Wrap the response in a structure with type and value
            response = {
                'type': 'dataframe',
                'value': response_json
            }
            return response
        else:
            # Wrap other types of responses as well, you may need to handle other data types
            response = {
                'type': 'other',
                'value': result.get('value', 'No data')
            }
            return jsonify(response), 200
        
    def format_plot(self, result):
        if 'value' in result:
            image_path = result['value']
            if os.path.exists(image_path):
                # Instead of directly serving the file, return the path within the desired structure
                response = {
                    'type': 'plot',
                    'value': image_path
                }
                return response
            else:
                return jsonify({'error': 'Image file not found'}), 404
        else:
            return jsonify({'error': 'Invalid result structure for plot'}), 400
        
    def format_other(self, result):
        if 'value' in result:
            try:
                # Wrap the serializable value
                response = {
                    'type': 'other',
                    'value': result['value']
                }
                return response
            except TypeError:
                # If it's not directly serializable, convert to string and wrap
                response = {
                    'type': 'other',
                    'value': str(result['value'])
                }
                return response
        else:
            return jsonify({'error': 'Invalid result structure'}), 400
        
class UploadManager:
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder
        self.sessions = {}

    def get_sheet_names(self, file_path):
        try:
            xls = pd.ExcelFile(file_path)
            return xls.sheet_names
        except Exception as e:
            print(f"Error extracting sheet names: {e}")
            return []
        
    def allowed_file(self, filename, allowed_extensions):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions 

    def save_file(self, file):
        filename = secure_filename(file.filename)
        file_path = os.path.join(self.upload_folder, filename)
        file.save(file_path)
        return file_path, filename

upload_manager = UploadManager(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():
    user_id = request.args.get('user_id', 'anonymous')
    files = request.files.getlist('file')
    if not files:
        return jsonify({'error': 'No files provided'}), 400
    
    id = str(uuid.uuid4())  # Generate a unique ID
    
    uploaded_files_info = []
    for file in files:
        if upload_manager.allowed_file(file.filename, ALLOWED_EXTENSIONS):
            file_path, filename = upload_manager.save_file(file)
            sheets = upload_manager.get_sheet_names(file_path) if file.filename.endswith('.xlsx') else []
            uploaded_files_info.append({'filename': filename, 'file_path': file_path, 'sheets': sheets})
    
    if not uploaded_files_info:
        return jsonify({'error': 'No valid files uploaded'}), 400
    
    upload_manager.sessions[id] = {
        'user_id': user_id,
        'files': uploaded_files_info,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    return jsonify({
        'message': 'Files uploaded successfully',
        'id': id,
        'filenames': [info['filename'] for info in uploaded_files_info]
    }), 200

@app.route('/list-sheets', methods=['GET'])
def list_sheets():
    id = request.args.get('id')
    if id not in upload_manager.sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    files = upload_manager.sessions[id]['files']
    sheets_info = {file_info['filename']: file_info['sheets'] for file_info in files}
    
    return jsonify({'id': id, 'sheets_info': sheets_info}), 200

@app.route('/select-sheets', methods=['POST'])
def select_sheets():
    data = request.json
    id = data.get('id')
    selections = data.get('selections', [])
    
    if id not in upload_manager.sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session_data = upload_manager.sessions[id]

    for file_info in session_data['files']:
        for selection in selections:
            if file_info['filename'] == selection['filename']:
                file_info['selected_sheets'] = selection['sheets']

    return jsonify({
        'message': 'Sheets selected successfully',
        'id': id,
        'selections': selections
    }), 200

@app.route('/preview-selected-sheets', methods=['GET'])
def preview_selected_sheets():
    id = request.args.get('id')
    
    if id not in upload_manager.sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session_data = upload_manager.sessions[id]
    all_previews = []
    
    for file_info in session_data['files']:
        filename = file_info['filename']
        file_path = file_info['file_path']
        selected_sheets = file_info.get('selected_sheets', [])
        
        if not selected_sheets:
            continue
        
        file_previews = {'filename': filename, 'sheets': {}}
        
        try:
            if filename.endswith('.xlsx'):
                xls = pd.ExcelFile(file_path)
                for sheet_name in selected_sheets:
                    df = pd.read_excel(xls, sheet_name=sheet_name, nrows=5)
                    file_previews['sheets'][sheet_name] = df.to_dict(orient='records')
            elif filename.endswith('.csv') and 'csv_preview' in selected_sheets:
                df = pd.read_csv(file_path, nrows=5)
                file_previews['sheets']['csv_preview'] = df.to_dict(orient='records')
        except Exception as e:
            file_previews['error'] = f'Failed to read file or sheet: {str(e)}'
        
        all_previews.append(file_previews)
    
    return jsonify({'id': id, 'previews': all_previews}), 200

@app.route('/conversation', methods=['POST'])
def handle_conversation():
    data = request.json
    session_id = data.get('id')
    user_query = data.get('query')

    if not session_id or session_id not in upload_manager.sessions:
        return jsonify({'error': 'Invalid session ID'}), 404
    if not user_query:
        return jsonify({'error': 'Query not provided'}), 400

    # Retrieve the selected sheets for this session
    session_data = upload_manager.sessions[session_id]
    loaded_data = load_selected_sheets_data(session_data)

    # Process the user's query against the loaded data
    response = process_loaded_data_with_query(loaded_data, user_query)
    # Save the response in the session data
    upload_manager.sessions[session_id]['conversation_response'] = {'type':response['type'],'value':response['value']}
    return response

def process_loaded_data_with_query(loaded_data, user_query):
    try:
        # Check if there's only one DataFrame, indicating use of SmartDataframe
        if len(loaded_data) == 1:
            single_df = next(iter(loaded_data.values()))
            sdf = SmartDataframe(df=single_df, config={"llm": OpenAI(api_token=api_key), "save_charts": True,"save_charts_path": r"./Charts", "verbose": True, "response_parser": PandasDataFrame})
            response = sdf.chat(user_query)  # Process the query using SmartDataframe
        else:
            # Multiple DataFrames, indicating use of SmartDatalake
            sdl = SmartDatalake(data=list(loaded_data.values()), config={"llm": OpenAI(api_token=api_key), "save_charts": True,"save_charts_path": r"./Charts", "verbose": True, "response_parser": PandasDataFrame})
            response = sdl.chat(user_query)  # Process the query using SmartDatalake
        return response
    except Exception as e:
        return {'error': str(e)}

def load_selected_sheets_data(session_data):
    """
    Load data from selected sheets into pandas DataFrames.
    
    :param session_data: Data for the current session, containing file paths and selected sheets.
    :return: Dictionary of DataFrames, with each key being a unique identifier for the sheet.
    """
    loaded_data = {}
    
    for file_info in session_data['files']:
        file_path = file_info['file_path']
        selected_sheets = file_info.get('selected_sheets', [])
        
        for sheet_name in selected_sheets:
            # Handle Excel files
            if file_path.endswith('.xlsx'):
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    # Create a unique key for each sheet, e.g., "filename_sheetname"
                    key = f"{file_info['filename']}_{sheet_name}"
                    loaded_data[key] = df
                except Exception as e:
                    print(f"Failed to load sheet '{sheet_name}' from file '{file_path}': {e}")
            # Handle CSV files
            elif file_path.endswith('.csv') and sheet_name == 'default':  # Assuming 'default' for single-sheet CSV
                try:
                    df = pd.read_csv(file_path)
                    key = f"{file_info['filename']}"
                    loaded_data[key] = df
                except Exception as e:
                    print(f"Failed to load CSV file '{file_path}': {e}")
            else:
                print(f"Unsupported file type for '{file_path}'")

    return loaded_data

@app.route('/process-response/<session_id>', methods=['GET'])
def process_response(session_id):
    if session_id not in upload_manager.sessions or 'conversation_response' not in upload_manager.sessions[session_id]:
        return jsonify({'error': 'Invalid session ID or no conversation response found'}), 404

    data = upload_manager.sessions[session_id]['conversation_response']
    print(data)
    response_type = data.get('type')

    # Process plot responses
    if response_type == 'plot':
        image_path = data.get('value')
        if os.path.exists(image_path):
            filename = f"{session_id}_plot.png"
            return send_file(image_path, as_attachment=True)
        else:
            return jsonify({'error': 'Image file not found'}), 404

    # Process dataframe responses
    elif response_type == 'dataframe':
        df = pd.read_json(data['value'], orient='records')
        excel_file = to_excel(df)
        return send_excel_file(excel_file, f"{session_id}_data.xlsx")

    # Handle other types of responses
    else:
        return jsonify({'value': data.get('value')})
    
def to_excel(df):
    """
    Convert a DataFrame into a BytesIO Excel object to be used for downloading.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1')
        writer.save()
    output.seek(0)
    return output

def send_excel_file(excel_io, filename):
    """
    Send the Excel file stored in a BytesIO object as a download response, with a specified filename.
    """
    return Response(
        excel_io.getvalue(),
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
