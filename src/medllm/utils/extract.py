from drug_named_entity_recognition import find_drugs

def extract_drug(question):
    
    drugs = find_drugs(question.split(" "))
    drug_extract = set()
    print(drugs)
    for drug in drugs:
        drug_extract.add(drug[0]['name'])

    return list(drug_extract)

# usage
if __name__ == "__main__":

    question = "What is Saxenda vs Wegovy?"
    print(find_drugs("What is Ozempic".strip(" ")))
    print(extract_drug(question))

"""
Does Pepto-Bismol help with diarrhea? -> Can't find "bismuth subsalicylate"
What is Journavx for pain? -> Can't be found
"""