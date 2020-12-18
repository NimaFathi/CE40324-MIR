from src.information_retrieval_system import IRSystem
from src.QueryCorrection import correct_query


# print("pee pee")
op1 = int(input("Please select the language of the document:\n1. English\t2. Persian\n"))
while True:
    if op1 == 1:
        lang = 'english'
        break
    elif op1 == 2:
        lang = 'persian'
        break
    else:
        print("please enter a valid option (1 or 2)")
        op1 = int(input("1. English\t2. Persian\n"))

# op2 = int(input("\nWhich system you want to use?:\n1. information retrieval\n2. query correction\n"))
# flag1, flag2 = False, False
# while True:
#     if op2 == 1:
#         flag1 = True
#         break
#     elif op2 == 2:
#         flag2 = True
#         break
#     else:
#         print("please enter a valid option (1 or 2)")
#         op2 = int(input("1. information retrieval\n2. query correction\n"))

while True:
    ir = IRSystem(lang)
    query = input("\nenter your query:\n")
    op = input("\nquery correction? y/n: ")
    if op == 'y':
        ir.corrected_query(query)
    op = input("\ndeleted stop words? y/n: ")
    if op == 'y':
        ir.plot_stop_words()
    op = input("\ntf-idf search? y/n: ")
    if op == 'y':
        nwo = int(input("\nnumber of wanted outcomes: "))
        retrieve_type = input("\nretrieve type? body/ title: ")
        ir.retrieve_tfidf_answer(query, nwo, retrieve_type)
    op = input("\nproximity search? y/n: ")
    if op == 'y':
        window = int(input("\nwindow: "))
        retrieve_type = input("\nretrieve type? body/ title: ")
        ir.retrieve_proximity_answer(query, window, retrieve_type)
    rep = input("\ngo again? y/n: ")
    if rep == 'n':
        break
    # wanted_outcomes = int(input("\nenter the number of outcomes you want:"))
    # temp = int(input("\nwhat retrieve type you want to use?\n1. body\t2. title\n"))
    # if temp == 1:
    #     retrieve_type = "body"
    # else:
    #     retrieve_type = "title"
    # ir.retrieve_query_answer(query, wanted_outcomes, retrieve_type)
    # con = int(input("do you want to use query correction system? 1. yes 2. no\n"))
    # if con == 1:
    #     flag2 = True
    # else:
    #     break
