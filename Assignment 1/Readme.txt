How to run the program

For K-means
    $ python main.py kmeans [cluster-number] [clustering-Attributes-file] [training-size]
    Eg: $ python main.py kmeans 5 attributes.txt 375

    Generates cluster details

For KNN
    $ python main.py knn [k-number] [clustering-Attributes-file] [training-size]
    Eg: $ python main.py knn 3 small_attr.txt 375
    
    Generates Accuracy of the program 

Note: 
Sample clustering-Attributes-file s have been added and named as attributes.txt and also small_attr.txt
These files follow certain conventions. By following the conventions, the program could be run for 
any clustering or classification problem based on given attributes.

Referneces

https://pythonprogramming.net/reading-csv-files-python-3/
https://mail.python.org/pipermail/tutor/2002-March/013249.html
