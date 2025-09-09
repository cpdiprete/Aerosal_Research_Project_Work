# from typing import Union

# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# def read_root():
#     return {"Hellooo00900o90": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}

from nicegui import ui, events
import os
import opencvresearch

# def uploads(e:events.UploadEventArguments):
#     test = e.content.read().decode("utf-8")
#     print(test)
#     print(type(test))
mediatmpfolder = "media"

if not os.path.exists(mediatmpfolder):
    os.makedirs(mediatmpfolder)
    
def handle_upload(e:events.UploadEventArguments):
    video_path = os.path.join(mediatmpfolder, e.name)
    with open(video_path, "wb") as file:
        file.write(e.content.read())
    ui.notify(f'Uploaded {e.name}')
    opencvresearch.run_video_file(video_path)
    
    
# ui.upload(on_upload=uploads)
ui.upload(on_upload=handle_upload, 
        label="Upload Video", 
        on_rejected=lambda: ui.notify('Rejected!'), 
        max_file_size=5_000_000_000).classes('max-w-full')


# markdown= ui.markdown('Choose a video file')

ui.run()