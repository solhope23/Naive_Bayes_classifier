from model.trainer import Trainer
import pandas as pd
import pprint

def run_demo():


    df = pd.read_csv("buy_computer_data.csv")
    trainer = Trainer(df, 'Buy_Computer')
    t = trainer.train()
    pprint.pprint(t[0])
    print('\n')
    pprint.pprint(t[1])

if __name__ == '__main__':
    run_demo()