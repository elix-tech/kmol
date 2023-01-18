import os
import numpy as np
from io import StringIO
from datetime import datetime
from time import sleep
from functools import wraps
from pathlib import Path
import pytz

from boxsdk import JWTAuth, Client, BoxAPIException

from boxsdk import Client, user, folder, file
Folder = folder.Folder
User = user.User
File = file.File

from kmol.core.logger import LOGGER as logging
from ..configs import BoxConfiguration


def retry(ExceptionToCheck, tries=10, delay=10, logger=None):
    """Retry calling the decorated function using an exponential backoff.

    :param ExceptionToCheck: the exception to check. may be a tuple of
        exceptions to check
    :type ExceptionToCheck: Exception or tuple
    :type tries: int
    :param logger: logging.Logger instance: logger to use. If None, print
    """
    def deco_retry(f):

        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except ExceptionToCheck as e:
                    msg = "%s, Retrying in %d seconds..." % (str(e), mdelay)
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                    sleep(mdelay)
                    mtries -= 1
            return f(*args, **kwargs)

        return f_retry  # true decorator

    return deco_retry


class Box:
    def __init__(self, config: BoxConfiguration, role: str = "client"):
        mkdir = role == "server"
        self.cfg = config
        auth = JWTAuth.from_settings_file(self.cfg.box_configuration_path)
        self.client = Client(auth)
        self.user = self.client
        self.base_path = self.get_base_path(role)

        self.path_auth_dir = self.base_path / 'authentifications'
        self.path_hb_dir = self.base_path / 'heartbeats'
        self.auth_dir = self.get_folder_from_path(self.path_auth_dir, mkdir=mkdir)
        self.hb_dir = self.get_folder_from_path(self.path_hb_dir, mkdir=mkdir)

    @retry(BoxAPIException, logger=logging)
    def get_base_path(self, role):
        base_path = Path(self.cfg.shared_dir_name) / self.cfg.save_path
        if role == "server":
            date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            return base_path / date
        else:
            base_folder = self.get_folder_from_path(base_path)
            id_last = np.argmax([e.get().content_created_at for e in base_folder.get_items()])
            return base_path / list(base_folder.get_items())[id_last].name

    @retry(BoxAPIException, logger=logging)
    def set_user(self, name):
        users = self.get_users_with_names(name)
        if len(users) > 0:
            user = users[0]
        else:
            user = self.client.create_user(name, login=None)
        self.add_user_to_group(user)
        
        self.user = self.client.as_user(user) 
        self.auth_dir = self.auth_dir.as_user(user)
        self.hb_dir = self.hb_dir.as_user(user)

    @retry(BoxAPIException, logger=logging) 
    def add_user_to_group(self, user: User):
        groups = [g for g in self.client.get_groups(name=self.cfg.group_name)]
        if len(groups) > 0:
            # Should be only one group with this name
            group_id = groups[0].id
        if not self.is_user_in_group(user, group_id):
            self.client.group(group_id=group_id).add_member(user)
    
    @retry(BoxAPIException, logger=logging)
    def get_users_with_names(self, name):
        return [u for u in self.client.users(filter_term=name) if u.name == name]

    @retry(BoxAPIException, logger=logging)
    def is_user_in_group(self, user: User, group_id: str):
        is_same_group = lambda m: self.client.group_membership(m.id).get().group.id == group_id
        return np.any([is_same_group(m) for m in user.get_group_memberships()])

    @retry(BoxAPIException, logger=logging)
    def get_folders_from_name(self, folder: Folder, name, mkdir=False):
        folders = [i for i in folder.get_items() if i.name == name and type(i) == Folder]
        if len(folders) == 1:
            return folders[0]
        elif mkdir and len(folders) == 0:
            return folder.create_subfolder(name)
        else:
            raise FileNotFoundError(f"Missing folder '{name}' in folder '{folder}'")

    @retry(BoxAPIException, logger=logging)
    def get_folder_from_path(self, path:str, mkdir=False) -> Folder:
        folder = self.user.folder('0')
        for part in Path(path).parts:
            folder = self.get_folders_from_name(folder, part, mkdir=mkdir)
        return folder        

    @retry(BoxAPIException, logger=logging)
    def get_file_if_exist(self, folder: Folder, filename:str):
        files = [i for i in folder.get_items() if i.name == filename and type(i) == File]
        if len(files) == 1:
            return files[0]
        elif len(files) == 0:
            return None

    @retry(BoxAPIException, logger=logging)
    def upload_text(self, folder: Folder, file_name: str, text: str):
        stream = StringIO()
        stream.write(text)
        stream.seek(0)
        return folder.upload_stream(stream, file_name)

    @retry(BoxAPIException, logger=logging)
    def update_text(self, file: File, text: str):
        stream = StringIO()
        stream.write(text)
        stream.seek(0)
        return file.update_contents_with_stream(stream)

    @retry(BoxAPIException, logger=logging)
    def get_modify_time_file(self, file: File):
        change_time = datetime.strptime(file.content_modified_at, "%Y-%m-%dT%H:%M:%S%z")
        return change_time.astimezone(pytz.utc).timestamp()

    @retry(BoxAPIException, logger=logging)
    def get_last_hb_user(self, user: User):
        files = [f.get() for f in self.auth_dir.get_items() if f.get().created_by == user]
        assert len(files) == 1, "There should be only one heartbeat file for this user"
        return self.get_modify_time_file(files[0])

    @retry(BoxAPIException, logger=logging)
    def upload_file(self, filename: str, file_path: str, folder: Folder) -> None:
        """
        Upload a file to the given folder.
        Args:
            filename (str): The name of the file to upload.
            file_path (str): The path of the file to upload.
            folder_id (str, optional): The object of the folder to upload the file to.
        """
        a_file = folder.upload(file_path, file_name=filename)
        try:
            logging.info(f'File uploaded: {a_file.get()["name"]}')
        except:
            logging.error("Upload error")

    @retry(BoxAPIException, logger=logging)
    def count_checkpoints(self, folder_path: str) -> int:
        folder = self.get_folder_from_path(folder_path)
        files = folder.get_items(limit=100, offset=0)
        
        checkpoints_count = 0
        for file in files:
            if file.name.endswith(".pt"):
                checkpoints_count += 1

        return checkpoints_count

    @retry(BoxAPIException, logger=logging)
    def download_all_checkpoints(self, local_path: str, folder: Folder) -> None:
        """
        Download all files with the .pt extension from the given folder.
        Args:
            local_path (str): The local path to download the files to.
            folder_id (str, optional): The ID of the folder to download the files from. Defaults to '0' (the root folder).
        """
        files = folder.get_items(limit=100, offset=0)
        for file in files:
            if file.name.endswith(".pt"):
                self.download_file(file, local_path)

    @retry(BoxAPIException, logger=logging)
    def download_file(self, file: File, save_path: str):
        if not Path(save_path).exists():
            Path(save_path).mkdir(parents=True)
        with open(os.path.join(save_path, file.name), 'wb') as open_file:
            return self.user.file(file.id).download_to(open_file)
        