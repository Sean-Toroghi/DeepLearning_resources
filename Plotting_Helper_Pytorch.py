import numpy as np
# visualization
def viz_2(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    '''
    Plot decision boundary of a model predicting on X compare to y
    '''
    
    model.to("cpu")
    X,y = X.to("cpu"), y.to("cpu")
    
    #setup boundaries and grids
    x_min, x_max = X[:,0].min()-0.1 , X[:,0].max() + 0.1
    y_min, y_max = X[:,1].min()-0.1 , X[:,1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min,x_max, 10),
                        np.linspace(y_min, y_max))
    
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()
    
    model.eval()
    with torch.inference_mode():
        y_logit = model(X_to_pred_on)
    
    #predict label
    if len(torch.unique(y))>2: #multiclass classification
        y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
    else: # binary classification
        y_pred = torch.round(torch.sigmoid(y_logit))
        
    #plot
    y_pred=  y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx,yy,y_pred, camp= plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap = plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.min())
    plt.ylim((yy.min()), yy.min())
    plt.show();