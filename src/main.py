# from src.infromation_retrieval_system import IRSystem
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

op2 = int(input("\nWhich system you want to use?:\n1. information retrieval\n2. query correction\n"))
flag1, flag2 = False, False
while True:
    if op2 == 1:
        flag1 = True
        break
    elif op2 == 2:
        flag2 = True
        break
    else:
        print("please enter a valid option (1 or 2)")
        op2 = int(input("1. information retrieval\n2. query correction\n"))

# if flag1:
#     ir = IRSystem(lang)
#     query = input("\nnow enter your query:\n")
#     wanted_outcomes = int(input("\nenter the number of outcomes you want:"))
#     temp = int(input("\nwhat retrieve type you want to use?\n1. body\t2. title\n"))
#     if temp == 1:
#         retrieve_type = "body"
#     else:
#         retrieve_type = "title"
#     ir.retrieve_query_answer(query, wanted_outcomes, retrieve_type)

if flag2:
    query = input("enter your misspelled query:\n")
    dictionary = input("now enter a dictionary containing correct form of your query's words "
                       "(and maybe other words):\n")
    print("corrected query is:\n")
    print(correct_query(query, dictionary.split(), 0.1))
