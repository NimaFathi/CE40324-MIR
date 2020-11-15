from src.infromation_retrieval_system import IRSystem
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

query = 0
while True:
    if flag1:
        ir = IRSystem(lang)
        query = input("\nnow enter your query:\n")
        wanted_outcomes = int(input("\nenter the number of outcomes you want:"))
        temp = int(input("\nwhat retrieve type you want to use?\n1. body\t2. title\n"))
        if temp == 1:
            retrieve_type = "body"
        else:
            retrieve_type = "title"
        ir.retrieve_query_answer(query, wanted_outcomes, retrieve_type)
        con = int(input("do you want to use query correction system? 1. yes 2. no\n"))
        if con == 1:
            flag2 = True
        else:
            break

    if flag2:
        if query == 0:
            query = input("enter your misspelled query:\n")
        else:
            new_query = int(input("do you want to use a new query or the last one?\n1. new 2. previous\n"))
            if new_query == 1:
                query = input("enter your misspelled query:\n")

        dictionary = input("now enter a dictionary containing correct form of your query's words "
                           "(and maybe other words):\n")
        print("corrected query is:\n")
        print(correct_query(query, dictionary.split()))
        break
