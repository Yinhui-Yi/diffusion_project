import requests

url = "http://10.123.58.113:5000/upload"  # 替换为服务器IP或域名
file_path = "image.jpg"  # 要上传的图片路径

try:
    with open(file_path, 'rb') as f:
        files = {'file': (file_path.split('/')[-1], f, 'image/jpeg')}
        print(files)
        response = requests.post(url, files=files)
    
    print(response.json())  # 打印服务器返回的JSON
except Exception as e:
    print(f"Error: {e}")