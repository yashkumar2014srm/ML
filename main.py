import pandas as pd
from flask import Flask, render_template
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/file')
def compute():
    file = r'train.csv'
    df = pd.read_csv(file)
    df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis=1, inplace=True)
    df['GarageCond'].fillna(df['GarageCond'].mode()[0], inplace=True)
    df['GarageType'].fillna(df['GarageType'].mode()[0], inplace=True)
    df['GarageQual'].fillna(df['GarageQual'].mode()[0], inplace=True)
    df['GarageYrBlt'].fillna(df['GarageYrBlt'].mode()[0], inplace=True)
    df['GarageFinish'].fillna(df['GarageFinish'].mode()[0], inplace=True)
    df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0], inplace=True)
    df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0], inplace=True)
    df['BsmtQual'].fillna(df['BsmtQual'].mode()[0], inplace=True)
    df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0], inplace=True)
    df['BsmtCond'].fillna(df['BsmtCond'].mode()[0], inplace=True)
    df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0], inplace=True)
    df['MasVnrType'].fillna(df['MasVnrType'].mode()[0], inplace=True)
    df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)
    df[['OverallQual', 'OverallCond']] = df[['OverallQual', 'OverallCond']].astype(str)
    df['YearBuilt_diff'] = df['YrSold'] - df['YearBuilt']
    df['YearRemodAdd_diff'] = df['YrSold'] - df['YearRemodAdd']
    df['GarageYrBlt_diff'] = df['YrSold'] - df['GarageYrBlt']
    df.drop(['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold', 'MoSold'], axis=1, inplace=True)
    df_1 = df.select_dtypes(include=['float64', 'int64'])
    df_1 = df
    df_1.drop(['Id'], axis=1, inplace=True)
    X = df_1
    y = X.pop('SalePrice')
    df_categ = X.select_dtypes(include=['object'])
    df_dummies = pd.get_dummies(df_categ, drop_first=True)
    X = X.drop(list(df_categ.columns), axis=1)
    X = pd.concat([X, df_dummies], axis=1)
    scaler = StandardScaler()
    scaler.fit_transform(X)
    cols = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
    params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1,
                        0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0,
                        4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 50.0, 100.0,
                        500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]}
    ridge = Ridge()
    folds = 5
    folds = 5
    model_cv = GridSearchCV(estimator=ridge,
                            param_grid=params,
                            scoring='neg_mean_absolute_error',
                            cv=folds,
                            return_train_score=True,
                            verbose=1)
    model_cv.fit(X_train, y_train)
    cv_results = pd.DataFrame(model_cv.cv_results_)
    cv_results = cv_results[cv_results['param_alpha'] <= 1000]
    cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')
    y_pred = model_cv.predict(X_test)
    # input = pd.DataFrame(y_test)
    output = pd.DataFrame()
    output['Predicted'] = y_pred
    output['Actual'] = y_test.reset_index().drop(columns='index')
    #output.Columns = [['Predicted','Actual']]
    # input['Predict'] = output.reset_index().dr
    # merged_df = pd.merge(input,output)
    file1 = r'output/output.csv'
    output.to_csv(file1)
    # df.to_json(r'output/output.json', orient='records')
    return output.to_html(header="true", table_id="table")

if __name__ == '__main__':
    app.run(debug=True)
