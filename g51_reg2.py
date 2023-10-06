import numpy as np
from sklearn.linear_model import RidgeCV, LinearRegression, Lasso
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

def metrics(model, x, y, k, N):
    cv_results = cross_validate(model, x, y, cv = k, scoring = ('neg_mean_squared_error', 'r2', 'neg_mean_absolute_error'), return_train_score = True)
    SSE = abs(cv_results['test_neg_mean_squared_error'].mean()) 
    SAE = abs(cv_results['test_neg_mean_absolute_error'].mean())
    r2 = cv_results['train_r2'].mean()
    return SSE, SAE, r2

def display(models, SSE, SAE, r2):
    for i in range(len(SSE)):
        print("\n************", models[i], "************")
        print("SSE =", SSE[i])
        print("SAE =", SAE[i])
        print("r² =", r2[i])
    print("\n************ Conclusion ************")
    print("Best SSE is", min(SSE), "from", models[np.where(SSE == min(SSE))[0]])
    print("Best SAE is", min(SAE), "from", models[np.where(SAE == min(SAE))[0]])
    print("Best r² is", max(r2), "from", models[np.where(r2 == max(r2))[0]])


def main():
    x_train = np.load('../dados/reg2/Xtrain_Regression2.npy')
    y_train = np.load('../dados/reg2/Ytrain_Regression2.npy')
    x_test = np.load('../dados/reg2/Xtest_Regression2.npy')
    
    N = x_train.shape[0]
    k = 10

    SSE = []
    SAE = []
    r2 = []


    #####Outlier removal#####
    #Removes outliers until (norm of gradient) of linear predictor changes less than 20% from previous iteration
    x_train_fi = x_train
    y_train_fi = y_train
    aux1 = x_train
    aux2 = y_train
    grads = []
    
    print(f"{'          Element Index' : ^20}{'MSE' : ^20}{'Difference of Grads' : ^30}")
    
    for i in range(N):
        
        lin = LinearRegression().fit(aux1, aux2)
        grads.append(lin.coef_)
        
        if i == 0:
            print(f"[{i}]{'N/A' : ^20}{'N/A' :^20}{'N/A' : ^30}")
        else:
            if (np.linalg.norm(grads[i] - grads[i - 1]) < 0.2 * np.linalg.norm(grads[i - 1] - grads[i - 2])): # Exceeded the treshold
                #plt.xlabel(r"$i$", fontsize=14)
                #plt.ylabel(r"$(y_i - \hat{y}_i)^2$", fontsize=14)
                #plt.title(f"After all removals ({i-1})", fontsize=14)
                #plt.plot(e)
                #plt.show()
                print(f"\nNumber of outliers: {i-1}\n")
                N_new=i-1
                break 
                
        x_train_fi = aux1
        y_train_fi = aux2
        y_pred = lin.predict(x_train_fi)
        distances = np.abs(y_pred - y_train_fi)
        
        if i == 0:
            #plt.title(f"Starting point", fontsize=14)
            pass
        else:
            #plt.title(f"After removing {i} outliers", fontsize=14)
            print(f"[{i}]{idx : ^20}{np.sum(e) / len(e) : >20}{np.linalg.norm(grads[i] - grads[i - 1]) : ^30}")
        #plt.xlabel(r"$i$", fontsize=14)
        #plt.ylabel(r"$(y_i - \hat{y}_i)^2$", fontsize=14)
        
        idx = np.argmax(distances)
        e = (y_pred - y_train_fi) ** 2
            

        #plt.plot(e, zorder = 1)
        #plt.plot(idx, e[idx], "o", color = 'r', zorder = 2)
        #plt.show()
        aux1 = np.delete(arr=aux1, obj=idx, axis=0)
        aux2 = np.delete(arr=aux2, obj=idx, axis=0)
 


    #####Feature removal#####
    # Removing redundant/irrelevant features, i.e, the ones that, if removed, make the
    # sse score go down.
    indexes_to_remove=[]
    null_coefs=[]
    regr = LinearRegression().fit(x_train_fi, y_train_fi)
    SSE_baseline = metrics(regr, x_train_fi, y_train_fi, k, N)
    for dim in range(x_train_fi.shape[1]):
        x_new = np.delete(arr = x_train_fi, obj = dim, axis = 1)
        regr_new = LinearRegression().fit(x_new, y_train_fi)
        SSE_new = metrics(regr_new, x_new, y_train_fi, k, N)
        if SSE_new < SSE_baseline:
            indexes_to_remove.append(dim)
            print(f"Eliminated feature {dim}")
            null_coefs.append(0)
        else:
            null_coefs.append(1)
    x_train_removed_features = np.delete(arr = x_train_fi, obj = indexes_to_remove, axis = 1)
    x_test = np.delete(arr = x_test, obj = indexes_to_remove, axis = 1)

    #####BEST MODEL#####
    ####################
    alphas_r = np.arange(0.0001, 0.01, 0.00001)
    ridgecv = RidgeCV(alphas = alphas_r, cv = k).fit(x_train_removed_features, y_train_fi.ravel())
    print('Best alpha RidgeCV: ', ridgecv.alpha_)
    SSE = metrics(ridgecv, x_train_removed_features, y_train_fi.ravel(), k, N_new)[0]
    print(f"SSE = {SSE}")


    #Final prediction
    y_pred = ridgecv.predict(x_test).reshape(-1,1)  
    print(y_pred.shape)   
    np.save('Ytest_Regression2', y_pred)
    

if __name__ == "__main__":
    main()