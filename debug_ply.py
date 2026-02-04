import os

def analyze_ply(filepath):
    print(f"Analyzing {filepath}")
    try:
        file_size = os.path.getsize(filepath)
    except FileNotFoundError:
        print("File not found.")
        return

    print(f"Total File Size: {file_size}")
    
    with open(filepath, 'rb') as f:
        content = f.read()
        
    header_end_marker = b"end_header"
    header_end_idx = content.find(header_end_marker)
    
    if header_end_idx == -1:
        print("Error: No 'end_header' found.")
        return

    # The header ends after 'end_header\n' or 'end_header\r\n'
    # strict ply: end_header\n
    ptr = header_end_idx + len(header_end_marker)
    while ptr < len(content) and content[ptr] in [13, 10]: # \r or \n
        ptr += 1
        
    header_size = ptr
    print(f"Header Size: {header_size}")
    
    payload_size = file_size - header_size
    print(f"Actual Payload Size: {payload_size}")
    
    # Parse Header
    header_text = content[:header_end_idx].decode('ascii', errors='ignore')
    lines = header_text.split('\n')
    
    vertex_count = 0
    face_count = 0
    vertex_props = 0
    
    current_element = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
            current_element = "vertex"
        elif line.startswith("element face"):
            face_count = int(line.split()[-1])
            current_element = "face"
        elif line.startswith("property") and current_element == "vertex":
            vertex_props += 1
            
    print(f"Vertices: {vertex_count}, Props: {vertex_props} (assuming float32=4bytes)")
    print(f"Faces: {face_count}")
    
    # Calculate Expected Size
    # Vertices: count * props * 4 bytes
    expected_vertex_data = vertex_count * vertex_props * 4
    print(f"Expected Vertex Data: {expected_vertex_data}")
    
    # Use the actual payload to see what remains for faces
    remaining_for_faces = payload_size - expected_vertex_data
    print(f"Remaining for faces: {remaining_for_faces}")
    
    if remaining_for_faces < 0:
        print("CRITICAL: Payload too small even for vertices!")
        diff = expected_vertex_data - payload_size
        print(f"Missing {diff} bytes for vertices.")
    
    elif face_count > 0:
        bytes_per_face = remaining_for_faces / face_count
        print(f"Avg bytes per face: {bytes_per_face}")
        # Triangle face list uchar uint: 1 + 3*4 = 13 bytes
        # Quad face list uchar uint: 1 + 4*4 = 17 bytes
        if abs(bytes_per_face - 13.0) < 0.01:
            print("Matches Triangles (13 bytes/face)")
            print("File seems OK size-wise for triangles.")
        elif abs(bytes_per_face - 17.0) < 0.01:
            print("Matches Quads (17 bytes/face)")
            print("File seems OK size-wise for quads.")
        else:
            print(f"Does not match standard Triangle(13) or Quad(17). Mismatch: {bytes_per_face}")
    
    else:
        print("No faces declared.")
        if remaining_for_faces > 0:
            print(f"Warning: {remaining_for_faces} extra bytes at end of file.")
        elif remaining_for_faces == 0:
            print("Perfect match for point cloud.")

if __name__ == "__main__":
    print("--- Bowling.ply ---")
    analyze_ply("/root/workspace/taichi_dataset/meshes/Bowling/Bowling.ply")
    print("\n--- pillow.ply ---")
    analyze_ply("/root/workspace/taichi_dataset/meshes/Pillow/pillow.ply")
