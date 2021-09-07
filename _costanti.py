
_CITIES = {}

_CITIES['africa'] = ['kinshasa', 'lagos', 'cairo', 'giza', 'johannesburg', 'rabat', 'casablanca', 'dakar', 
                    'nairobi', 'capetown', 'tripoli']

_CITIES['asia'] = ['tokyo', 'delhi', 'shanghai', 'dhaka', 'mumbai', 'beijing', 'osaka', 'seoul', 'istanbul', 
                  'islamabad', 'bangkok', 'singapore', 'jakarta', 'dubai', 'baghdad', 'kabul', 'pyongyang', 
                  'ulaanbaatar', 'kathmandu'] 

_CITIES['australia'] = ['melbourne', 'sydney', 'perth', 'wellington']

_CITIES['europe'] = ['london', 'cardiff', 'belfast', 'edinburgh', 'dublin', 'birmingham', 'leeds', 'glasgow', 
                    'sheffield', 'berlin', 'hamburg', 'munich', 'frankfurt', 'stuttgart', 'rome', 'milan', 
                    'florence', 'naples', 'turin', 'messina', 'bologna', 'madrid', 'barcelona', 
                    'paris', 'lyon', 'marseille', 'toulouse', 'moscow', 'saint_petersburg', 'krakow', 
                    'warsaw', 'lisbon', 'porto', 'amsterdam', 'stockholm', 'belgrade', 'vienna', 'oslo', 
                    'budapest', 'copenhagen', 'athens', 'helsinki', 'tallinn', 'kyiv', 'prague', 'brussels', 
                    'tirana', 'bucharest', 'vilnius', 'riga', 'reykjavik', 'bratislava']

_CITIES['north_america'] = ['new_york', 'los_angeles', 'chicago', 'houston', 'phoenix', 'philadelphia', 
                           'san_antonio', 'san_diego', 'dallas', 'san_jose', 'austin', 'jacksonville', 
                           'fort_worth', 'indianapolis', 'columbus', 'charlotte', 'san_francisco', 'seattle', 
                           'denver', 'washington', 'nashville', 'el_paso', 'boston', 'portland', 'detroit', 
                           'memphis', 'kansas_city', 'new_orleans', 'miami', 'honolulu', 'ottawa', 'toronto', 
                           'montreal', 'vancouver', 'havana', 'la_vegas']

_CITIES['south_america'] = ['bogota', 'caracas', 'lima', 'brasilia', 'quito', 'montevideo', 'buenos_aires', 
                           'la_paz', 'sao_paulo', 'santiago_chile', 'rio_janeiro']

_TOT_CITIES = [item for sublist in list(_CITIES.values()) for item in sublist] 



# -----------------------------


_TOPIC = {}
_TOPIC['crime'] = ['abduction', 'arson', 'assassination', 'assault', 'bigamy', 'blackmail', 
                 'bombing', 'bribery', 'burglary', 'child_abuse', 'corruption', 'crime', 'cybercrime', 
                 'domestic_violence', 'drunk_driving', 'embezzlement', 'espionage', 'forgery', 
                 'fraud', 'genocide', 'hijacking', 'homicide', 'hooliganism', 'kidnapping', 
                 'libel', 'looting', 'lynching', 'manslaughter', 'mugging', 'murder', 'perjury', 
                 'poaching', 'rape', 'riot', 'robbery', 'pickpocket', 'shoplifting', 'slander', 
                  'smuggling', 'speeding', 'terrorism', 'theft', 'trafficking', 'treason', 
                  'trespassing', 'vandalism']

_TOPIC['education'] = ['education', 'school', 'high_school', 'university', 'student', 'professor', 
                      'teach', 'college', 'schooling', 'graduate', 'undergraduate', 'undergrad']

_TOPIC['food'] = ['food', 'restaurant', 'cuisine', 'chef', 'gourmet', 'meal', 'cooking', 'recipe', 'culinary', 'diner', 
                 'lunch', 'dinner', 'cafe', 'bistro']

_TOPIC['work'] = ['job', 'work', 'salary', 'business', 'career', 'office', 'secretary', 'colleague', 'profession']

_TOPIC['nightlife'] = ['nightlife', 'pub', 'bar', 'disco', 'club', 'cinema', 'movie', 'festival', 'nightclub']

_TOPIC['cold'] = ['cold', 'snow', 'freeze', 'icy']

_TOPIC['hot'] = ['hot', 'sun', 'sweat', 'torrid', 'warm']

_TOPIC['transportation'] = ['car', 'helicopter', 'taxi', 'plane', 'airplane', 'bus', 'bike', 'skateboard', 
                           'bicycle', 'subway']

_TOPIC['environment'] = ['ecology', 'sustainability', 'organic', 'renewable', 'ecological', 'biodegradable', 'recyclable']

_TOPIC['pollution'] = ['pollution', 'waste', 'toxic', 'toxin', 'impurity', 'greenhouse', 'contaminant']

_TOPIC['history'] = ['king', 'queen', 'historical', 'castle', 'ruin', 'church', 'philosophy', 
                    'medieval', 'history', 'knight', 'museum', 'temple']

_TOPIC['gambling'] = ['gambling', 'game', 'luck', 'chance', 'card', 'roulette', 'casino', 
                     'blackjack', 'poker', 'slot', 'gambler', 'betting', 'wagering', 'croupier']

_TOPIC['alcohol'] = ['alcohol', 'alcoholic', 'binge_drinking', 'drunk', 'drunkenness', 'liquor', 'beer', 
                    'mojito', 'cocktail', 'rum', 'vodka', 'tequila', 'gin', 'whisky', 'whiskey', 'brandy', 
                    'scotch', 'bourbon', 'champagne', 'booze', 'alcoholic_beverage', 'alcohol_consumption']

_TOPIC['music'] = ['dance', 'music', 'jazz', 'reggaeton', 'song', 'melody', 'musical', 'reggae', 'hip_hop', 
                  'rap', 'rapper', 'rock_roll', 'lyrics', 'ballad', 'pop', 'k_pop']

_TOPIC['fitness'] = ['fitness', 'health', 'healthy', 'aerobics', 'workout', 'cardio', 'jog', 'jogging', 'climbing', 
                    'trekking', 'weight_training', 'strength_training', 'hike']

_TOPIC['vegetation'] = ['vegetation', 'tree', 'forest', 'rainforest', 'grassland', 'fir', 'pine', 'oak', 'acre', 
                       'field', 'garden', 'flower']

_TOPIC['animals'] = ['safari', 'zoo', 'animal', 'wild', 'wildlife', 'aquarium', 'fauna', 'reptile', 'mammal']

_TOPIC['sport'] = ['soccer', 'football', 'basketball', 'volleyball', 'golf', 'baseball', 'hockey', 'bowling', 
                  'ping_pong', 'tennis', 'swimming', 'boxing', 'skiing', 'rugby', 'surfing', 'karate', 
                   'snowboarding', 'judo', 'badminton']

_TOPIC['professions'] = ['baker' , 'butcher', 'cook', 'farmer', 'fireman', 'gardener', 'hairdresser' , 'journalist', 
                        'lawyer', 'mason' , 'mechanic', 'plumber', 'policeman' , 'postman', 'singer', 'soldier', 
                        'taxi_driver', 'teacher', 'waiter', 'pilot', 'engineer', 'doctor', 'nurse', 'goldsmith', 
                        'actor', 'tailor']


_TOT_TOPIC = [item for sublist in list(_TOPIC.values()) for item in sublist]

