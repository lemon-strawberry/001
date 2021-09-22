import requests

line_notify_token = ''
line_notify_api = 'https://notify-api.line.me/api/notify'


def line_notify(message, file_name=None):
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}
    if file_name == None:
        try:
            requests.post(line_notify_api, data=payload, headers=headers)
        except:
            pass
    else:
        try:
            files = {"imageFile": open(file_name, "rb")}
            requests.post(line_notify_api, data=payload, headers=headers, files=files)
        except:
            pass
