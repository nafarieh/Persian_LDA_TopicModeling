# TopicModeling

1- pip install -r requirements.txt

2-python LDA_Topicmodeling.py

3- uvicorn LDA_Topicmodeling:app --reload

4- curl --location --request POST '127.0.0.1:8000/' --header 'Content-Type: application/json' --data-raw '{"filename": "Instagram_comment_server.csv","topic_num": 5}'

