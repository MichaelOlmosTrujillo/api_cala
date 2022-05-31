1. To check the data from scrapping run file scrapping.py
2. To check models from steps 3 and 4 run analisis.py file
3. Data from scrapping was save in df_mi_claro.csv file
4. Data from step 3 was save in vaders_content.csv file
5. To check the api run file api_cala.py
6. Deploy in heroku use the URL: http://api-cala.herokuapp.com/items/20
   where we could replace the number 20 by any positive integer. This will 
   show the 20 first comments with their respective positive, negative and 
   neutral connotation
7. The model from step 3 use the library nltk for natural language processing
   to give the positive, negative, neutral and compound values for each 
   comment
8. The model for step 4 use a random forest classification algorithm.

   