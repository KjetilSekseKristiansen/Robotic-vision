import numpy as np
Alder=int(input("Hvor gammel er du?: "))
Bodd=int(input("Hvor lenge har du bodd i Tulleby?: "))
if Alder>=30 and Bodd>=9:
    print("Du kan bli Ordfører eller sitte i bystyret")
elif Alder>=30:
    if (Bodd<9 and Bodd>=5):
        print("Du kan sitte i bystyret")
        print("Prøv igjen om " + str(9-Bodd) + " år for å bli ordfører")
elif Alder < 30 and Alder>=25:
    if 9-Bodd<=0:
        print("Du kan sitte i bystyret")
        print("Prøv igjen om " + str(30-Alder) + " år for å bli ordfører")
elif Alder<=25 and Bodd>5:
    print("Du er ikke kvalifisert enda, prøv igjen om " + str(25-Alder) + "år")
elif Alder<3:
    print('young fucker')
else:
    print('fuck off cunt')
print(np.maximum(30-Alder,5-Bodd))