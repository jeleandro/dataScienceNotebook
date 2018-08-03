def ks_binary(x,y, dtype=np.float32):
    notNAN= np.isnan(x)==False;
    y = y[notNAN];
    x = x[notNAN]; 
    
    argsorted = x.argsort();
    
    cdfFalse = (y[argsorted]==False).cumsum(dtype=dtype);
    cdfFalse /= cdfFalse[-1];

    cdfTrue = (y[argsorted]).cumsum(dtype=dtype);
    cdfTrue /= cdfTrue[-1];
    
    ks = np.abs(cdfFalse - cdfTrue).max();
    ks_mean = np.abs(cdfFalse - cdfTrue).mean();
    ks_stab = np.mean((cdfFalse - cdfTrue) >0);
    if ks_stab != 1 and ks_stab != 0:
        ks_entropy = - (ks_stab*np.log(ks_stab) + (1-ks_stab)*np.log(1-ks_stab));
    else:
        ks_entropy = 0;
    return ks, ks_entropy, ks_mean, argsorted, cdfFalse, cdfTrue;
    
    
def histKS(var, truth, cutoff=2.5,bins=30,colors=['#d7191c', '#2c7bb6'],alfa=0.5, labels=['False','True'], density=False):
    notNAN= np.isnan(var)==False;
    truth=truth[notNAN];
    var = var[notNAN];
    
    var = np.clip(var,a_min=np.percentile(var,cutoff), a_max=np.percentile(var,100-cutoff));
    plt.xlim((np.percentile(var,cutoff), np.percentile(var,100-cutoff)))
    plt.hist(var[truth == False], label=labels[0], bins=bins, color=colors[0], alpha=alfa);
    plt.hist(var[truth], label=labels[1], bins=bins,color=colors[1], alpha=alfa);
    
    ax2 = plt.twinx()
    
    ks, ks_entropy,ks_mean, argsorted, cdfFalse, cdfTrue = ks_binary(var, truth)
    ax2.plot(var[argsorted],cdfFalse,color=colors[0], label=labels[0])
    ax2.plot(var[argsorted],cdfTrue,color=colors[1], label=labels[1])
    ax2.set_ylim((0.0,1.05))

    plt.xlabel(u"Distância");
    plt.legend(loc='best')
    return ks, ks_entropy, ks_mean;
    
    
def plotKSROC(y_pred, y_true):
    plt.figure(figsize=(10,3))
    plt.subplot(1,2,1);
    ks =histKS(y_pred, y_true);
    plt.title("Kolmogorov-Simirnov %0.2f | %0.2f | %0.2f" % (ks[0], ks[1], ks[2]))
    plt.subplot(1,2,2);
    fpr, tpr, thresholds = roc_curve(y_score=y_pred,y_true=y_true)
    plt.plot(fpr, tpr, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % auc(fpr,tpr));
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
