```
probability = day_profile[probability]
consumption_range = day_profile[consumption_range]

random_chance = random_float(0 to 1)

if probability > random_chance:
    drink = True
else:
    drink = False

servings = 0
if drink:
    servings = random_int(consumption_range)
```
