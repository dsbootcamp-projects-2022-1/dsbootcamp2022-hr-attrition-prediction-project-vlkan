from fastapi import FastAPI
from ml import predict
from preprocess import preprocess
from sample import Sample

import uvicorn



app = FastAPI()


@app.post("/predict/")
def read_items(sample: Sample) -> int:
    sample_dict = sample.__dict__
    preprocessed_sample = preprocess(sample_dict)
    prediction = predict(preprocessed_sample)

    return prediction


@app.get("/whoami")
def whoami() -> str:
    # TODO
    isim = "Volkan"
    soyisim = "Önder"
    mail = "thkvolkanonder@gmail.com"
    
    person_card = {
        "isim": isim,
        "soyisim": soyisim,
        "mail": mail
    }

    return person_card


@app.get("/model_card")
def model_card() -> str:
    # TODO

    model_card = {
        'model_name': 'logisticReg',
        'model_description': '',
        'model_version': '1.1',
        'model_author': 'Volkan Önder',
        'model_author_mail': 'thkvolkanonder@gmail.com',
        'model_creation_date': '31.03.2022',
        'model_last_update_date': '03.04.2022',
        'required_parameters_list': '',
        'required_parameters_descriptions': 'Logistic Regresion, solver=liblinear, max_iter=1000, C=0.1',
    }

    return model_card


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, log_level="info")





