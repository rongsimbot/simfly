import urllib.request, time, os

os.makedirs("/tmp/simfly_web/phase15", exist_ok=True)

# Capture a frame from the MJPEG stream
url = "http://192.168.1.199:8080/video_feed"
try:
    # Read a few frames and save the last one
    response = urllib.request.urlopen(url, timeout=10)
    boundary = b'--frame'
    data = b''
    frames = []
    chunk = response.read(500000)  # Read up to 500KB
    response.close()
    
    # Extract the last JPEG frame
    parts = chunk.split(boundary)
    for part in parts:
        if b'Content-Type: image/jpeg' in part:
            jpeg_start = part.find(b'\xff\xd8')
            jpeg_end = part.rfind(b'\xff\xd9')
            if jpeg_start >= 0 and jpeg_end > jpeg_start:
                jpeg_data = part[jpeg_start:jpeg_end+2]
                frames.append(jpeg_data)
    
    if frames:
        with open("/tmp/simfly_web/phase15/dashboard_frame.jpg", "wb") as f:
            f.write(frames[-1])
        print(f"Captured frame ({len(frames[-1])} bytes) - saved as dashboard_frame.jpg")
    else:
        print("No JPEG frames found in stream")
        print(f"Read {len(chunk)} bytes, {len(parts)} parts")
        # Show first 200 bytes of first part for debugging
        for i, p in enumerate(parts[:3]):
            print(f"Part {i}: starts with {p[:50]}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
