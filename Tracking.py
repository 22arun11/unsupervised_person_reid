import streamlit as st
import cv2
import face_recognition as frg
import yaml 
import numpy as np
import hashlib
from utils import recognize, build_dataset, get_info_from_id, get_databse
# Path: code\app.py

# Initialize session state for admin login
if 'admin_logged_in' not in st.session_state:
    st.session_state.admin_logged_in = False

st.set_page_config(layout="wide")
#Config
cfg = yaml.load(open('config.yaml','r'),Loader=yaml.FullLoader)
PICTURE_PROMPT = cfg['INFO']['PICTURE_PROMPT']
WEBCAM_PROMPT = cfg['INFO']['WEBCAM_PROMPT']

st.sidebar.title("Settings")

# Add a Database option that requires authentication
menu = ["Webcam", "Picture", "Database"]
choice = st.sidebar.selectbox("Input type", menu)

#Put slide to adjust tolerance (only for Webcam and Picture)
if choice != "Database":
    TOLERANCE = st.sidebar.slider("Tolerance",0.0,1.0,0.5,0.01)
    st.sidebar.info("Tolerance is the threshold for face recognition. The lower the tolerance, the more strict the face recognition. The higher the tolerance, the more loose the face recognition.")


# Function to get top N matches
def get_top_matches(face_encoding, n=3, tolerance=0.6):
    database = get_databse()
    matches = []
    
    # Extract all encodings and corresponding IDs from database
    for idx, entry in database.items():
        if 'encoding' in entry:
            db_encoding = entry['encoding']
            # Calculate face distance
            distance = frg.face_distance([db_encoding], face_encoding)[0]
            # Only consider matches below the tolerance threshold
            if distance <= tolerance:
                matches.append((idx, distance, entry))
    
    # Sort by distance (smaller distance = better match)
    matches.sort(key=lambda x: x[1])
    
    # Return top N matches
    return matches[:n]

# Function to delete an entry from the database
def delete_entry(idx):
    # Load the database
    import pickle as pkl
    from utils import PKL_PATH
    
    # Open and get the current database
    with open(PKL_PATH, 'rb') as f:
        database = pkl.load(f)
    
    # Remove the entry
    if idx in database:
        del database[idx]
        
        # Save the updated database
        with open(PKL_PATH, 'wb') as f:
            pkl.dump(database, f)
        return True
    return False

# Database Section (requires authentication)
if choice == "Database":
    st.title("Database Management")
    
    # Show login form if not logged in
    if not st.session_state.admin_logged_in:
        with st.form("admin_login"):
            st.subheader("Admin Authentication Required")
            st.warning("You must log in as an administrator to access database information.")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                # Authentication with specified credentials
                if username == "admin" and password == "reid@123":
                    st.session_state.admin_logged_in = True
                    st.experimental_rerun()
                else:
                    st.error("Invalid credentials")
    
    # Show database management tools after successful login
    if st.session_state.admin_logged_in:
        st.success("Authenticated as Administrator")
        
        # Create tabs for different database functions
        db_tabs = st.tabs(["Database Overview", "Search", "Management"])
        
        # Database Overview tab
        with db_tabs[0]:
            st.subheader("Database Overview")
            
            # Database statistics
            try:
                database = get_databse()
                st.info(f"Current database has {len(database)} entries")
                
                # Show database entries with images
                st.write("Database Entries:")
                
                if len(database) > 0:
                    # Create a more visual display with images and delete buttons
                    for idx, entry in database.items():
                        with st.expander(f"ID: {entry.get('id', 'Unknown')} - Name: {entry.get('name', 'Unknown')}"):
                            # Create two columns - one for image, one for info and delete button
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                if 'image' in entry:
                                    st.image(entry['image'], caption="Person Image", width=150)
                                else:
                                    st.warning("No image available")
                            
                            with col2:
                                st.write(f"**Index:** {idx}")
                                st.write(f"**ID:** {entry.get('id', 'Unknown')}")
                                st.write(f"**Name:** {entry.get('name', 'Unknown')}")
                                if 'timestamp' in entry:
                                    # Format timestamp nicely
                                    from datetime import datetime
                                    timestamp = datetime.fromtimestamp(entry['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
                                    st.write(f"**Added:** {timestamp}")
                                
                                # Delete button with confirmation
                                delete_key = f"delete_{idx}"
                                if st.button("Delete Entry", key=delete_key):
                                    st.warning(f"Are you sure you want to delete entry {entry.get('id', 'Unknown')}?")
                                    confirm_key = f"confirm_{idx}"
                                    if st.button("Yes, Delete", key=confirm_key):
                                        if delete_entry(idx):
                                            st.success("Entry deleted successfully")
                                            st.experimental_rerun()
                                        else:
                                            st.error("Failed to delete entry")
                else:
                    st.warning("No entries found in database")
                    
            except Exception as e:
                st.error(f"Error accessing database: {str(e)}")
        
        # Search tab
        with db_tabs[1]:
            st.subheader("Search Database")
            
            # Option to view specific entry details
            entry_id = st.text_input("Enter ID to search")
            if entry_id and st.button("Search"):
                try:
                    database = get_databse()
                    found = False
                    for idx, entry in database.items():
                        if entry.get('id') == entry_id:
                            found = True
                            st.success(f"Found entry with ID: {entry_id}")
                            
                            # Show image and details side by side
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.image(entry.get('image'), caption=f"Image for {entry.get('name', 'Unknown')}")
                            
                            with col2:
                                st.write(f"**Name:** {entry.get('name', 'Unknown')}")
                                st.write(f"**Index:** {idx}")
                                if 'timestamp' in entry:
                                    from datetime import datetime
                                    timestamp = datetime.fromtimestamp(entry['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
                                    st.write(f"**Added:** {timestamp}")
                                
                                # Delete button with confirmation
                                if st.button("Delete Entry", key=f"search_delete_{idx}"):
                                    st.warning(f"Are you sure you want to delete entry {entry_id}?")
                                    if st.button("Yes, Delete", key=f"search_confirm_{idx}"):
                                        if delete_entry(idx):
                                            st.success("Entry deleted successfully")
                                            st.experimental_rerun()
                                        else:
                                            st.error("Failed to delete entry")
                            
                            break
                    
                    if not found:
                        st.error(f"No entry found with ID: {entry_id}")
                except Exception as e:
                    st.error(f"Error searching database: {str(e)}")
        
        # Management tab
        with db_tabs[2]:
            st.subheader("Database Management")
            
            # Option to rebuild database
            if st.button("REBUILD DATASET"):
                with st.spinner("Rebuilding dataset..."):
                    build_dataset()
                st.success("Dataset has been reset")
            
            # Option to add new admin (demonstration only)
            with st.expander("Admin Management"):
                st.write("Manage administrator accounts")
                
                with st.form("add_admin"):
                    st.write("Add New Admin")
                    new_username = st.text_input("Username")
                    new_password = st.text_input("Password", type="password")
                    confirm_password = st.text_input("Confirm Password", type="password")
                    add_submit = st.form_submit_button("Add Admin")
                    
                    if add_submit:
                        if not new_username or not new_password:
                            st.error("Username and password required")
                        elif new_password != confirm_password:
                            st.error("Passwords don't match")
                        else:
                            st.success(f"Admin {new_username} added successfully")
        
        # Logout button
        if st.button("Logout"):
            st.session_state.admin_logged_in = False
            st.experimental_rerun()

# Picture Mode
elif choice == "Picture":
    st.title("Face Recognition App")
    st.write(PICTURE_PROMPT)
    
    # Create two columns - one for input, one for output
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_images = st.file_uploader("Upload", type=['jpg','png','jpeg'], accept_multiple_files=True)
        if len(uploaded_images) != 0:
            # Read uploaded image with face_recognition
            for image in uploaded_images:
                image_data = frg.load_image_file(image)
                processed_image, name, id = recognize(image_data, TOLERANCE) 
                
                # Display the processed input image with bounding boxes
                st.image(processed_image, caption="Uploaded Image with Detection")
                
                # Get face encodings from the image
                face_locations = frg.face_locations(image_data)
                if face_locations:
                    face_encodings = frg.face_encodings(image_data, face_locations)
                    if face_encodings:
                        # Get top 3 matches for the first detected face
                        top_matches = get_top_matches(face_encodings[0], n=3, tolerance=TOLERANCE)
                        
                        # Display top matches in the second column
                        with col2:
                            if top_matches:
                                st.subheader("Top Matches")
                                match_cols = st.columns(min(3, len(top_matches)))
                                
                                for i, (idx, distance, entry) in enumerate(top_matches):
                                    with match_cols[i]:
                                        st.image(entry['image'], caption=f"Match #{i+1}\nDistance: {distance:.2f}")
                            else:
                                st.info("No matches found")
                    else:
                        with col2:
                            st.info("No face encodings could be generated")
                else:
                    with col2:
                        st.info("No faces detected")
        else:
            col2.info("Please upload an image")

# Webcam Mode  
# In your Tracking.py file, modify the webcam section:

elif choice == "Webcam":
    st.title("Face Recognition App")
    st.write(WEBCAM_PROMPT)
    
    # Create two columns - one for webcam, one for reference images
    col1, col2 = st.columns(2)
    
    # Camera Settings
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    with col1:
        FRAME_WINDOW = st.image([])
        tracking_status = st.empty()  # Status placeholder
        st.write("Live Feed with Detection")
    
    with col2:
        # Create 3 smaller columns for top matches
        st.write("Top Matches from Database")
        match_cols = st.columns(3)
        match_containers = [col.empty() for col in match_cols]
    
    # Store last face encoding to avoid redundant matching
    last_encoding = None
    
    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to capture frame from camera")
            st.info("Please turn off the other app that is using the camera and restart app")
            st.stop()
        
        # Create a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Get face locations and encodings
        face_locations = frg.face_locations(frame)
        
        # Draw rectangles and labels directly instead of using recognize()
        if face_locations:
            face_encodings = frg.face_encodings(frame, face_locations)
            
            # Show tracking status
            tracking_status.success("Person Identified")
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Draw rectangle
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Only add "Person Identified" text, no name or ID
                cv2.putText(display_frame, "Person Identified", (left, top-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                
                # If we have an encoding, show distance and find matches
                if len(face_encodings) > 0:
                    current_encoding = face_encodings[0]
                    
                    # Only update matches if face encoding has changed significantly
                    if last_encoding is None or frg.face_distance([last_encoding], current_encoding)[0] > 0.1:
                        last_encoding = current_encoding
                        
                        # Get top 3 matches
                        top_matches = get_top_matches(current_encoding, n=3, tolerance=TOLERANCE)
                        
                        # Display top matches
                        if top_matches:
                            for i in range(3):
                                if i < len(top_matches):
                                    idx, distance, entry = top_matches[i]
                                    match_containers[i].image(entry['image'], 
                                                             caption=f"Match #{i+1}\nDistance: {distance:.2f}")
                                    
                                    # If this is the closest match, show the distance on the frame
                                    if i == 0:
                                        distance_text = f"Distance: {distance:.2f}"
                                        cv2.putText(display_frame, distance_text, (left, top-40), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                                else:
                                    match_containers[i].empty()
                        else:
                            for container in match_containers:
                                container.info("No match found")
        else:
            # No face detected
            tracking_status.empty()
            for container in match_containers:
                container.info("No face detected")
        
        # Display the processed frame
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(rgb_frame)