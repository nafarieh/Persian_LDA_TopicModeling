from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from LDA_Topicmodeling import run, read_uploaded_files


app = FastAPI()


@app.get("/")
def index():
    return {"status": "ok"}


@app.post("/uploadfile/", response_class=FileResponse)
async def create_upload_file(topic_num: int, data_file: UploadFile = File(...)):
# async def create_upload_file(data_file: UploadFile = File(...)):
    df, stopwords = read_uploaded_files(data_file=data_file)
    case_list = run(df=df, stopwords=stopwords, num_topic=topic_num)
    with open("cases.txt", "w") as f:
        print(case_list, file=f)
    return "./word_cloud.png"
