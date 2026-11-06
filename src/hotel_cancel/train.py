from hotel_cancel import config
from hotel_cancel import data
from hotel_cancel import evaluate
from hotel_cancel import features



def main():
    #Step1: Load the raw data
    df = data.load_raw(config.DATA_PATH)

    #step2: Clean the data
    df = data.drop_unused_columns(df)
    df = data.add_dates(df)

    #step3: Split the data into train, validation, and test sets
    train_df,val_df,test_df = data.temporal_split(df)

    #step4: Split the data into features and target
    X_train, y_train = data.split_xy(train_df)
    X_val, y_val = data.split_xy(val_df)
    X_test, y_test = data.split_xy(test_df)

    #step5: Build the preprocessor
    preprocesor = evaluate.build_preprocessor(X_train)

   


if __name__ == "__main__":
    main()