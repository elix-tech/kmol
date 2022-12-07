import numpy as np
from io import StringIO
from datetime import datetime
from boxsdk import Client, user, folder, file
Folder = folder.Folder
User = user.User
File = file.File
from pathlib import Path
import pytz

from boxsdk import JWTAuth
from boxsdk import Client

from ...configs import BoxConfiguration

class Box:
    def __init__(self, config: BoxConfiguration):
        self.cfg = config
        auth = JWTAuth.from_settings_file(self.cfg.box_configuration_path)
        self.client = Client(auth)
        self.user = self.client

        self.path_auth_dir = Path(self.cfg.shared_dir_name) / self.cfg.save_path / 'authentifications'
        self.path_hb_dir = Path(self._cfg_bbox.shared_dir_name) / self._cfg_bbox.save_path / 'heartbeats'
        self.auth_dir = self.get_folder_from_path(self.path_auth_dir, mkdir=True)
        self.hb_dir = self.get_folder_from_path(self.path_auth_dir, mkdir=True)

    def set_user(self, name):
        users = self.get_users_with_names(name)
        if len(users) > 0:
            user = users[0]
        else:
            user = self.client.create_user(name, login=None)
        self.user = self.client.as_user(user) 
        self.auth_dir = self.auth_dir.as_user(user)
        self.hb_dir = self.hb_dir.as_user(user)
    
    def get_users_with_names(self, name):
        return [u for u in self.client.users(filter_term=name) if u.name == name]

    def is_user_in_group(self, user: User, group_id: str):
        is_same_group = lambda m: self.client.group_membership(m.id).get().group.id == group_id
        return np.any([is_same_group(m) for m in user.get_group_memberships()])

    def get_folders_from_name(self, folder: Folder, name, mkdir=False):
        folders = [i for i in folder.get_items() if i.name == name and type(i) == Folder]
        if len(folders) == 1:
            return folders[0]
        elif mkdir and len(folders) == 0:
            return folder.create_subfolder(name)
        else:
            raise FileNotFoundError(f"Missing folder '{name}' in folder '{folder.name}'")

    def get_folder_from_path(self, path:str, mkdir=False) -> Folder:
        folder = self.client.folder('0')
        for part in Path(path).parts:
            folder = self.get_folders_from_name(folder, part, mkdir=mkdir)
        return folder        

    def get_file_if_exist(self, folder: folder.Folder, filename:str):
        files = [i for i in folder.get_items() if i.name == filename and type(i) == File]
        if len(files) == 1:
            return files[0]
        elif len(files) == 0:
            return None

    def upload_text(self, folder: Folder, file_name: str, text: str):
        stream = StringIO()
        stream.write(text)
        stream.seek(0)
        return folder.upload_stream(stream, file_name)

    def update_text(self, file: File, text: str):
        stream = StringIO()
        stream.write(text)
        stream.seek(0)
        return file.update_contents_with_stream(stream)

    def get_modify_time_file(self, file):
        change_time = datetime.strptime(file.content_modified_at, "%Y-%m-%dT%H:%M:%S%z")
        return change_time.astimezone(pytz.utc).timestamp()

    def get_last_hb_user(self, user):
        files = [f for f in self.bbox.auth_dir.get_items() if f.created_by == user]
        assert len(files) == 1, "There should be only one heartbeat file for this user"
        return self.get_modify_time_file(files[0])



# def get_users_with_names(client: Client, name):
#     return [u for u in client.users(filter_term=name) if u.name == name]


# def is_user_in_group(client: Client, user: User, group_id: str):
#     is_same_group = lambda m: client.group_membership(m.id).get().group.id == group_id
#     return np.any([is_same_group(m) for m in user.get_group_memberships()])

# def get_folders_from_name(folder: Folder, name):
#     folders = [i for i in folder.get_items() if i.name == name and type(i) == Folder]
#     assert len(folders) == 1, f"Missing or duplicate element with name '{name}' \
#         in folder '{folder.name}'"
#     return folders[0]

# def get_folder_from_path(client: Client, path:str) -> Folder:
#     folder = client.folder('0')
#     for part in Path(path).parts:
#         folder = get_folders_from_name(folder, part)
#     return folder

# def get_file_if_exist(folder: folder.Folder, filename:str):
#     files = [i for i in folder.get_items() if i.name == filename and type(i) == File]
#     if len(files) == 1:
#         return files[0]
#     elif len(files) == 0:
#         return None

# def upload_text(folder: Folder, file_name: str, text: str):
#     stream = StringIO()
#     stream.write(text)
#     stream.seek(0)
#     return folder.upload_stream(stream, file_name)