from flask import Flask, render_template
import pandas as pd
import numpy as np
import os
from random import randint, random
import torch

app = Flask(__name__)


def load_items():
    ds_dir = './data/'
    ds = 'descriptions_stock.csv'
    path = os.path.join(ds_dir, ds)
    data = pd.read_csv(path)
    return data


def create_cart(items):
    n = randint(2, 15)
    mask = np.random.choice(items.shape[0], n, replace=False)
    return items['Description'].iloc[mask], items['StockCode'].iloc[mask]


def __suggest(items, codes, model):
    all_codes = pd.DataFrame(data=items['StockCode'].unique(),
                             columns=['StockCode'])
    cart = torch.tensor(np.array(all_codes.isin(list(codes)),
                                 dtype=np.float64).flatten()).double()
    sigma = torch.nn.Sigmoid()
    result = sigma(model(cart)).detach().numpy()

    result[result >= 0.5001] = 1
    result[result < 0.5001] = 0
    indices = np.where(result != 0)[0]
    if len(indices) == 0:
        n = randint(1, 6)
        mask = np.random.choice(items.shape[0], n, replace=False)
        return items['Description'].iloc[mask]
    return list((items[items['StockCode'].isin(all_codes.iloc[indices].to_numpy().flatten())][
        'Description']).to_numpy())[:6]


def get_suggestions(items, codes, model):
    if random() < 0.5:
        n = randint(1, 6)
        mask = np.random.choice(items.shape[0], n, replace=False)
        return items['Description'].iloc[mask], "Baseline"
    else:
        return __suggest(items, codes, model), "Novel"


@app.route("/")
def main():
    model = torch.jit.load('./weights/model.pt').double()
    items = load_items()
    cart, codes = create_cart(items)
    suggestions, source = get_suggestions(items, codes, model)
    return render_template('index.html', items=cart, suggestions=suggestions)
