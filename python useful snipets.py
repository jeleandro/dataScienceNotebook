#string comparison
def LevenshteinDistance(stext, ttext):
    # for all i and j, d[i,j] will hold the Levenshtein distance between
    # the first i characters of s and the first j characters of t
    # note that d has (m+1)*(n+1) values
    m = len(stext);
    n = len(ttext);
    
    d =np.zeros((len(stext), len(ttext)),dtype=np.int32);

 
    # source prefixes can be transformed into empty string by
    # dropping all characters
    for i in range(m):
        d[i, 0] = i;
    
    # target prefixes can be reached from empty source prefix
    # by inserting every character
    for j in range(n):
        d[0, j] = j;
    
    for j in range(n):
        for i in range(m):
            if stext[i] == ttext[j]:
                substitutionCost = 0
            else:
                substitutionCost = 1
            d[i, j] = min(d[i-1, j] + 1,                   # deletion
                             d[i, j-1] + 1,                   # insertion
                             d[i-1, j-1] + substitutionCost)  # substitution
        #plt.imshow(d)
 
    #return d[m-1,n-1];
   
    return d;