import pandas as pd

countries = pd.read_csv('A01_Pandas/countries.csv')

print('1. ¿A que continente pertenece Tunisia?')

answer_1 = countries[(countries.country == 'Tunisia')]
print('Tunisia pertenece al continente de',answer_1['continent'].head(1).values[0])

print('2. ¿En que paises la esperanza de vida fue mayor a 80 en el 2007?')

answer_2 = countries[(countries.lifeExp > 80) & (countries.year == 2007)]
print('Los paises son:',answer_2['country'].values[0:len(answer_2)])

print('3. ¿Que pais de America tiene el mayor PIB?')

answer_3 = countries[(countries.continent == 'Americas')].sort_values('gdpPercap', ascending=False)
print('EL pais de america con mayor PIB es',answer_3['country'].head(1).values[0])

print('4. ¿Que pais tenia mas habitantes en 1967 entre Venezuela y Paraguay?')

answer_4 = countries[(countries.country == 'Venezuela') | (countries.country == 'Paraguay') & (countries.year == 1967)].sort_values('pop', ascending=False)
print(answer_4['country'].head(1).values[0], 'fue el pais con mas habitantes en 1967.')

print('5. ¿En que año Panama alcanzo una esperanza de vida mayor a 60 años?')

answer_5 = countries[(countries.country == 'Panama') & (countries.lifeExp > 60)].sort_values('lifeExp', ascending=False)
print('En el año',answer_5['year'].head(1).values[0])

print('6. ¿Cual es el promedio de la esperanza de vida en Africa en 2007?')

answer_6 = countries[(countries.continent == 'Africa') & (countries.year == 2007)]
print('EL promedio es de ',sum(answer_6['lifeExp'])/len(answer_6))

print('7. ¿Enlista los paises en que el PIB de 2007 fue menor que su PIB en 2002?')

pib_2007 = countries[(countries.year == 2007)]
pib_2002 = countries[(countries.year == 2002)]
answer_7 = ""
for count in range(len(pib_2007)):
    if pib_2007.iat[count,5] < pib_2002.iat[count,5]:
        answer_7 = pib_2007.iat[count,0] if answer_7 == "" else answer_7 + ', ' + pib_2007.iat[count,0]

print('Los paises son:', answer_7)

print('8. ¿Que pais tiene mas habitantes en 2007?')

answer_8 = countries[(countries.year == 2007 )].sort_values('pop', ascending=False)
print('El pais con mas habitantes es:',answer_8['country'].head(1).values[0])

print('9. ¿Cuantos habitantes tiene America en 2007?')

answer_9 = countries[(countries.continent == 'Americas') & (countries.year == 2007)]
print('El total de habitantes es: ', sum(answer_9['pop']))

print('10. ¿Que continente tiene menos habitantes en 2007?')

answer_10 = countries[(countries.year == 2007)].sort_values('pop')
print('El continente de', answer_10['continent'].head(1).values[0])

print('11. ¿Cual es el promedio de PIB en Europa?')

answer_11 = countries[(countries.continent == 'Europe')]
print('El promedio es de', sum(answer_11['gdpPercap'])/len(answer_11))

print('12. Cual fue el primer pais de Europa en superar los 70 millones de habitantes')

answer_12 = countries[(countries.continent == 'Europe') & (countries['pop'] > 70e6)].sort_values('year')
print('El primer pais de Europa en superar los 70 millones de habitantes fue',countries['country'].head(1).values[0], 'en el año', countries['year'].head(1).values[0])