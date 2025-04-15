from flask import Blueprint, request, jsonify
import embedding.addFile as addFile
import embedding.updateFile as updateFile
import embedding.deleteFile as deleteFile
import embedding.search as search

NO_FILENAME = "No filename part in the request"
NO_VALID_FILENAME = "No filename provided"

embedding_bp = Blueprint('embedding_bp', __name__)


@embedding_bp.route('/add-files', methods=['POST'])
def add_files():
    
    print("Adding files endpoint")
    data = request.get_json()
    
    if 'filenames' not in data:
        return jsonify({"error": "No filenames part in the request"}), 400
    
    filenames = data.get('filenames')
    if not filenames or filenames == []:
        return jsonify({"error": "No filenames provided"}), 400

    addFile.add_files(filenames)
    return jsonify({"message": "Files added successfully"}), 200
    
    
@embedding_bp.route('/update-file', methods=['POST'])
def update_file():
    
    print("Updating file endpoint")
    data = request.get_json()
    
    if 'filename' not in data:
        return jsonify({"error": NO_FILENAME}), 400
    
    if 'newFilename' not in data:
        return jsonify({"error": "No new filename part in the request"}), 400
    
    filename = data.get('filename')
    if not filename or filename == "":
        return jsonify({"error": NO_VALID_FILENAME}), 400
    
    new_filename = data.get('newFilename')
    if not new_filename or new_filename == "":
        return jsonify({"error": "No new filename provided"}), 400

    updateFile.update_file(filename, new_filename)
    return jsonify({"message": "File updated successfully"}), 200
    
    
@embedding_bp.route('/delete-file', methods=['POST'])
def delete_file():
    
    print("Deleting file endpoint")
    data = request.get_json()
    
    if 'filename' not in data:
        return jsonify({"error": NO_FILENAME}), 400
    
    filename = data.get('filename')
    if not filename or filename == "":
        return jsonify({"error": NO_VALID_FILENAME}), 400

    deleteFile.delete_file(filename)
    return jsonify({"message": "File deleted successfully"}), 200


@embedding_bp.route('/search', methods=['POST'])
def search_endpoint():
    
    print("Search endpoint")
    data = request.get_json()
    
    if 'filename' not in data:
        return jsonify({"error": NO_FILENAME}), 400
    
    filename = data.get('filename')
    if not filename or filename == "":
        return jsonify({"error": NO_VALID_FILENAME}), 400

    if 'count' not in data:
        count = 3
    else:
        count = data.get('count')

    return jsonify(search.find_similar_images(filename, count)), 200
