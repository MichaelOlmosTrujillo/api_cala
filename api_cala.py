# -*- coding: utf-8 -*-
"""
Created on Sun May 29 07:54:37 2022

@author: miolmos
"""
from fastapi import FastAPI
import uvicorn
from analisis import vaders_content

app = FastAPI()

@app.get('/')
def index():
    return {"message":"Prueba Cala Data science"}

@app.get("/items/{q}")
def read_items(q:int):
    ents = []
    vaders_content_show = vaders_content[:q]
    print(vaders_content_show[:q])
    for index, row in vaders_content_show.iterrows():
        if index < q:
            ents.append({"content": row['content'],
                    "negativa": row['neg'],
                    "positiva": row['pos'],
                    "compound": row['compound']})
    return {"message": 'showing {} rows'.format(q), "rows": ents}
    
# def get_name(name:str):
#    return {'Welcome To Krish Youtube Channel': f'{name}'}

if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)