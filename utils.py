import face_recognition as frg
import pickle as pkl 
import os 
import cv2 
import numpy as np
import yaml
from collections import defaultdict

information = defaultdict(dict)
cfg = yaml.load(open('config.yaml','r'),Loader=yaml.FullLoader)
DATASET_DIR = cfg['PATH']['DATASET_DIR']
PKL_PATH = cfg['PATH']['PKL_PATH']

def get_databse():
    with open(PKL_PATH,'rb') as f:
        database = pkl.load(f)
    return database
def recognize(image, TOLERANCE): 
    database = get_databse()
    
    # Check if database is empty or not properly formatted
    if not database:
        return image, 'Unknown', 'Unknown'
    
    # Get encodings from database - handle different database structures
    known_encoding = []
    db_keys = list(database.keys())
    db_items = []
    
    for idx in db_keys:
        try:
            if 'encoding' in database[idx]:
                known_encoding.append(database[idx]['encoding'])
                db_items.append((idx, database[idx]))
        except Exception as e:
            print(f"Error accessing database item {idx}: {e}")
    
    # If no valid encodings found, return unknown
    if not known_encoding:
        return image, 'Unknown', 'Unknown'
    
    name = 'Unknown'
    id = 'Unknown'
    face_locations = frg.face_locations(image)
    face_encodings = frg.face_encodings(image, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = frg.compare_faces(known_encoding, face_encoding, tolerance=TOLERANCE)
        distance = frg.face_distance(known_encoding, face_encoding)
        name = 'Unknown'
        id = 'Unknown'
        
        if True in matches:
            match_index = matches.index(True)
            
            # Safely access the database item
            try:
                db_idx, db_item = db_items[match_index]
                name = db_item.get('name', 'Unknown')
                id = db_item.get('id', 'Unknown')
                distance = round(distance[match_index], 2)
                cv2.putText(image, str(distance), (left, top-30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error retrieving match info: {e}")
        
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    
    return image, name, id
def isFaceExists(image): 
    face_location = frg.face_locations(image)
    if len(face_location) == 0:
        return False
    return True
def submitNew(name, id, image, old_idx=None):
    database = get_databse()
    #Read image 
    if type(image) != np.ndarray:
        image = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)

    isFaceInPic = isFaceExists(image)
    if not isFaceInPic:
        return -1
    
    #Encode image
    encoding = frg.face_encodings(image)[0]
    
    #Update mode 
    if old_idx is not None: 
        new_idx = old_idx
    #Add mode
    else:
        # Generate a unique key for this entry
        # Combine ID with timestamp to ensure uniqueness
        import time
        new_idx = f"{id}_{int(time.time())}"
        
        # Check if we need to prevent duplicate IDs
        # If you want to allow multiple images per ID, comment this block out
        existing_id = [database[i]['id'] for i in database.keys()]
        if id in existing_id and not allow_multiple_per_id:
            return 0
    
    # Convert image to RGB for consistent storage
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Store in database with the new unique index
    database[new_idx] = {
        'image': image,
        'id': id, 
        'name': name,
        'encoding': encoding,
        'timestamp': time.time(),  # Add timestamp for tracking
        'index_in_sequence': get_sequence_count(database, id)  # Track which image this is for this ID
    }
    
    # Save to file
    with open(PKL_PATH, 'wb') as f:
        pkl.dump(database, f)
    return True

def get_sequence_count(database, id):
    """Count how many images already exist for this ID"""
    count = 0
    for key in database:
        if database[key].get('id') == id:
            count += 1
    return count
def get_info_from_id(id): 
    database = get_databse() 
    for idx, person in database.items(): 
        if person['id'] == id: 
            name = person['name']
            image = person['image']
            return name, image, idx       
    return None, None, None
def deleteOne(id):
    database = get_databse()
    id = str(id)
    for key, person in database.items():
        if person['id'] == id:
            del database[key]
            break
    with open(PKL_PATH,'wb') as f:
        pkl.dump(database,f)
    return True
def build_dataset():
    counter = 0
    for image in os.listdir(DATASET_DIR):
        image_path = os.path.join(DATASET_DIR,image)
        image_name = image.split('.')[0]
        parsed_name = image_name.split('_')
        person_id = parsed_name[0]
        person_name = ' '.join(parsed_name[1:])
        if not image_path.endswith('.jpg'):
            continue
        image = frg.load_image_file(image_path)
        information[counter]['image'] = image 
        information[counter]['id'] = person_id
        information[counter]['name'] = person_name
        information[counter]['encoding'] = frg.face_encodings(image)[0]
        counter += 1

    with open(os.path.join(DATASET_DIR,'database.pkl'),'wb') as f:
        pkl.dump(information,f)

if __name__ == "__main__": 
    deleteOne(4)

