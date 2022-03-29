from fastapi import FastAPI
from ml import predict
from preprocess import preprocess
from sample import Sample


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
    isim = "JOHN"
    soyisim = "DOE"
    mail = "mail@mail.com"
    
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
        'model_name': '',
        'model_description': '',
        'model_version': '',
        'model_author': '',
        'model_author_mail': '',
        'model_creation_date': '',
        'model_last_update_date': '',
        'required_parameters_list': '',
        'required_parameters_descriptions': '',
    }

    return model_card








