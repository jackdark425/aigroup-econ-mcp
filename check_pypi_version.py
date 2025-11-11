import urllib.request
import json

def get_latest_version():
    try:
        url = 'https://pypi.org/pypi/aigroup-econ-mcp/json'
        response = urllib.request.urlopen(url)
        data = json.loads(response.read())
        latest_version = data["info"]["version"]
        print(f"PyPI上aigroup-econ-mcp的最新版本: {latest_version}")
        return latest_version
    except Exception as e:
        print(f"查询失败: {e}")
        return None

if __name__ == "__main__":
    get_latest_version()