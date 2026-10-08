_Generated from `vocabulary.tsv` and `class_names.tsv` by `python scripts/stimuli/vocabulary.py --readme`. Edit those files, not this page._

# Word lists

`vocabulary.tsv` is the lexical vocabulary: 1,481 words and names and 80 pseudowords in eight lists, one row per word. `class_names.tsv` is the curation record of the ImageNet-1k class names, one row per candidate with its decision; 200 are kept. `two_clue_pilot.tsv` and `two_clue_confirmatory.tsv` hold the two-clue items.

No fingerprint covers the roles, pairs or categories in `vocabulary.tsv` and `class_names.tsv`, so a cache built before an edit to either file still loads after it. After an edit, rerun every stage from `scripts/run_stimuli.sh` on.

## Reading the vocabulary

```python
from src.stimuli.vocabulary import read_vocabulary

vocabulary = read_vocabulary()
# or, without the checks and renames:
import pandas as pd
vocabulary = pd.read_csv("lexicon/vocabulary.tsv", sep="\t", dtype=str,
                         keep_default_na=False)
```

| Column | Content |
|---|---|
| `list` | The list the word belongs to, named by the slugs below |
| `item` | The first word of the word's item. An item of a relation list is one row of its table below; in the other lists an item is one word. |
| `item_role` | The word's part in its item. `spelling_twin` is the paper's one-edit word. A noun's role is `category`, a brand's `brand`, a pseudoword's `pseudoword`. |
| `word` | The rendered string: lowercase, three to ten letters, and unique in the file |
| `category` | The category of a noun or the industry of a brand |
| `spelling_cluster` | The spelling cluster of a noun or brand, 0 to 3. The derived tables call it `ortho`. |
| `concreteness` | Whether the item of list 1 is concrete or abstract. The derived tables call it `picturability`. |

The derived tables and the results name each pair relation as follows.

| Relation | List | Pair |
|---|---|---|
| `synonym` | Synonym and one-edit comparison | anchor and synonym |
| `spelling_twin` | Synonym and one-edit comparison | anchor and one-edit word |
| `antonym_synonym` | Synonym and antonym | anchor and synonym |
| `antonym` | Synonym and antonym | anchor and antonym |
| `negation` | Synonym and apparent negation/base | negation and synonym |
| `negation_base` | Synonym and apparent negation/base | negation and base |
| `singular_plural` | Singular and plural | singular and plural |
| `country_capital` | Country and capital | country and capital |

## The eight lists

Words counts distinct words, and Pairs counts pairs before the query direction is reversed.

| List | Items | Words | Pairs | Example and purpose |
|---|---|---|---|---|
| Synonym and one-edit comparison | 150 | 450 | 300 | torch-flashlight, torch-touch: meaning and spelling |
| Synonym and antonym | 50 | 150 | 100 | brave-bold, brave-cowardly: synonymy and opposition |
| Synonym and apparent negation/base | 35 | 105 | 70 | disappear-vanish, disappear-appear: meaning and similar form |
| Singular and plural | 40 | 80 | 40 | road-roads: form with related meaning |
| Country and capital | 60 | 120 | 60 | germany-berlin: factual association |
| Common nouns | 384 | 384 | - | Category membership |
| Brands | 192 | 192 | - | Industry membership |
| Pseudowords | 80 | 80 | - | Retrieval candidates and centering |
| Total | - | 1,561 | 570 |  |

### (1) Synonym and one-edit comparison

`synonym_one_edit`: 150 items, 450 words.

| Anchor | Synonym | One-edit word | Concreteness |
|---|---|---|---|
| torch | flashlight | touch | concrete |
| cooker | stove | booker | concrete |
| petrol | gasoline | patrol | concrete |
| purse | handbag | pulse | concrete |
| jug | pitcher | dug | concrete |
| cot | crib | colt | concrete |
| cart | trolley | dart | concrete |
| trench | ditch | wrench | concrete |
| quilt | blanket | guilt | concrete |
| outlet | socket | outset | concrete |
| chest | dresser | crest | concrete |
| loft | attic | soft | concrete |
| hallway | corridor | halfway | concrete |
| patio | terrace | ratio | concrete |
| rotary | roundabout | notary | concrete |
| couch | sofa | crouch | concrete |
| rug | carpet | bug | concrete |
| car | automobile | par | concrete |
| cap | hat | gap | concrete |
| cove | inlet | cone | concrete |
| coast | shore | toast | concrete |
| cloth | fabric | sloth | concrete |
| plank | board | blank | concrete |
| box | carton | bog | concrete |
| jar | pot | ear | concrete |
| bottle | flask | battle | concrete |
| hamper | basket | harper | concrete |
| cup | mug | cue | concrete |
| wire | cable | hire | concrete |
| mallet | hammer | ballet | concrete |
| corn | maize | morn | concrete |
| metro | subway | retro | concrete |
| shop | store | ship | concrete |
| gully | ravine | fully | concrete |
| perch | roost | peach | concrete |
| mitt | glove | mint | concrete |
| pad | guard | sad | concrete |
| pin | brooch | ain | concrete |
| candle | taper | handle | concrete |
| cape | cloak | care | concrete |
| bowl | basin | boil | concrete |
| bench | pew | bunch | concrete |
| robe | gown | roe | concrete |
| hut | shack | hue | concrete |
| fiddle | violin | riddle | concrete |
| fence | railing | hence | concrete |
| plain | prairie | pain | concrete |
| sea | ocean | tea | concrete |
| gravel | grit | travel | concrete |
| jet | airliner | jest | concrete |
| dock | wharf | mock | concrete |
| inn | hotel | ion | concrete |
| pub | tavern | cub | concrete |
| castle | fortress | cattle | concrete |
| diner | restaurant | liner | concrete |
| cafe | bistro | cave | concrete |
| toad | frog | load | concrete |
| broth | soup | both | concrete |
| jam | jelly | jaw | concrete |
| deck | porch | duck | concrete |
| cane | staff | sane | concrete |
| chute | slide | cute | concrete |
| map | chart | tap | concrete |
| boss | chief | loss | concrete |
| pal | friend | gal | concrete |
| aide | helper | abide | concrete |
| spy | agent | shy | concrete |
| boat | craft | bat | concrete |
| puddle | pool | paddle | concrete |
| cog | gear | log | concrete |
| pane | glass | plane | concrete |
| worm | grub | dorm | concrete |
| sill | ledge | dill | concrete |
| spout | nozzle | sprout | concrete |
| wool | fleece | fool | concrete |
| lift | elevator | lit | abstract |
| fog | mist | hog | abstract |
| breeze | wind | freeze | abstract |
| stem | stalk | seem | abstract |
| ring | band | wing | abstract |
| coin | token | chin | abstract |
| vat | tub | vet | abstract |
| pasture | meadow | posture | abstract |
| port | harbor | poet | abstract |
| branch | limb | brunch | abstract |
| yarn | thread | earn | abstract |
| brink | edge | blink | abstract |
| brood | litter | blood | abstract |
| pack | troop | puck | abstract |
| den | lair | din | abstract |
| buoy | float | busy | abstract |
| hatch | flap | hitch | abstract |
| meter | gauge | deter | abstract |
| path | track | math | abstract |
| haze | smog | maze | abstract |
| poster | flyer | potter | abstract |
| hike | trek | pike | abstract |
| feud | rift | fend | abstract |
| toil | labor | soil | abstract |
| pause | lull | cause | abstract |
| finding | discovery | funding | abstract |
| hobby | pastime | hubby | abstract |
| sleep | slumber | bleep | abstract |
| talk | speech | walk | abstract |
| debate | discussion | rebate | abstract |
| message | memo | massage | abstract |
| tale | story | table | abstract |
| start | onset | smart | abstract |
| warning | caution | warming | abstract |
| promise | pledge | premise | abstract |
| vacation | holiday | vocation | abstract |
| duty | obligation | duly | abstract |
| mission | assignment | fission | abstract |
| rumor | gossip | humor | abstract |
| sorrow | grief | borrow | abstract |
| worry | anxiety | sorry | abstract |
| gloom | melancholy | groom | abstract |
| scorn | contempt | sworn | abstract |
| awe | amazement | axe | abstract |
| hunch | instinct | lunch | abstract |
| loyalty | devotion | royalty | abstract |
| vigor | energy | rigor | abstract |
| tension | stress | pension | abstract |
| bulk | size | bunk | abstract |
| extent | scope | extant | abstract |
| era | epoch | err | abstract |
| myth | fallacy | moth | abstract |
| friction | discord | fraction | abstract |
| consent | permission | content | abstract |
| assent | approval | absent | abstract |
| denial | refusal | dental | abstract |
| ban | veto | lan | abstract |
| skill | expertise | skull | abstract |
| aptitude | competence | altitude | abstract |
| knack | talent | knock | abstract |
| flair | artistry | fair | abstract |
| fluke | accident | flake | abstract |
| risk | hazard | disk | abstract |
| danger | menace | ranger | abstract |
| big | large | bit | abstract |
| slight | minor | sight | abstract |
| wealth | prosperity | health | abstract |
| icy | frosty | icc | abstract |
| valiant | fearless | variant | abstract |
| bravely | boldly | gravely | abstract |
| real | actual | deal | abstract |
| extensive | widespread | expensive | abstract |
| heed | attention | held | abstract |
| fame | renown | tame | abstract |
| rank | status | bank | abstract |

### (2) Synonym and antonym

`synonym_antonym`: 50 items, 150 words.

| Anchor | Synonym | Antonym |
|---|---|---|
| brave | bold | cowardly |
| generous | giving | stingy |
| kind | gentle | cruel |
| patient | tolerant | restless |
| humble | modest | arrogant |
| polite | courteous | rude |
| cautious | careful | reckless |
| confident | assured | timid |
| orderly | neat | messy |
| important | vital | trivial |
| strong | mighty | weak |
| sincere | genuine | fake |
| ambitious | driven | complacent |
| eager | keen | reluctant |
| serious | solemn | playful |
| clumsy | awkward | nimble |
| elegant | refined | crude |
| simple | basic | complex |
| strict | rigid | lenient |
| energetic | lively | sluggish |
| excited | thrilled | bored |
| tough | resilient | fragile |
| precise | exact | sloppy |
| costly | pricey | cheap |
| first | initial | last |
| prior | earlier | later |
| crowded | packed | empty |
| tedious | boring | fun |
| fragrant | scented | smelly |
| outdated | obsolete | current |
| ordinary | average | special |
| sensible | rational | foolish |
| tasty | yummy | bland |
| abundant | ample | scarce |
| broken | damaged | intact |
| worn | shabby | flawless |
| brief | short | lengthy |
| glossy | polished | matte |
| damp | humid | arid |
| thorough | complete | careless |
| limited | finite | endless |
| hot | scorching | cold |
| huge | massive | tiny |
| wide | broad | narrow |
| bright | radiant | dim |
| old | ancient | new |
| towering | elevated | squat |
| luminous | glowing | murky |
| outlying | isolated | adjacent |
| scenic | pretty | dreary |

### (3) Synonym and apparent negation/base

`synonym_negation`: 35 items, 105 words.

| Negation | Synonym | Base |
|---|---|---|
| disappear | vanish | appear |
| disbelief | skepticism | belief |
| disclosure | revealing | closure |
| distract | deflect | tract |
| harmless | benign | harmful |
| illogical | absurd | logical |
| immoral | wicked | moral |
| impartial | neutral | partial |
| impending | imminent | pending |
| imperfect | flawed | perfect |
| incorrect | mistaken | correct |
| injury | harm | jury |
| invaluable | priceless | valuable |
| invent | formulate | vent |
| irregular | erratic | regular |
| nonviolent | peaceful | violent |
| unaware | ignorant | aware |
| uncertain | hesitant | certain |
| unclear | vague | clear |
| uncommon | rare | common |
| unequal | disparate | equal |
| uneven | bumpy | even |
| unfriendly | hostile | friendly |
| unlikely | doubtful | likely |
| unnatural | artificial | natural |
| unpaid | voluntary | paid |
| unpleasant | nasty | pleasant |
| unpopular | despised | popular |
| unsafe | risky | safe |
| unspoken | implicit | spoken |
| unstable | shaky | stable |
| untrue | false | true |
| unused | idle | used |
| unusually | remarkably | usually |
| unusual | weird | usual |

### (4) Singular and plural

`singular_plural`: 40 items, 80 words.

| Singular | Plural |
|---|---|
| road | roads |
| eye | eyes |
| station | stations |
| leader | leaders |
| iron | irons |
| heaven | heavens |
| bone | bones |
| egg | eggs |
| clock | clocks |
| shoe | shoes |
| nerve | nerves |
| taxi | taxis |
| barrel | barrels |
| organ | organs |
| cow | cows |
| sweat | sweats |
| trunk | trunks |
| archive | archives |
| cord | cords |
| marble | marbles |
| mod | mods |
| peanut | peanuts |
| mechanic | mechanics |
| drawer | drawers |
| viewer | viewers |
| prop | props |
| strait | straits |
| breaker | breakers |
| trumpet | trumpets |
| bun | buns |
| follower | followers |
| bowel | bowels |
| ala | alas |
| cracker | crackers |
| bead | beads |
| diver | divers |
| hoop | hoops |
| hippie | hippies |
| stair | stairs |
| clam | clams |

### (5) Country and capital

`country_capital`: 60 items, 120 words.

| Country | Capital |
|---|---|
| italy | rome |
| russia | moscow |
| germany | berlin |
| spain | madrid |
| india | delhi |
| japan | tokyo |
| china | beijing |
| ireland | dublin |
| belgium | brussels |
| austria | vienna |
| korea | seoul |
| greece | athens |
| thailand | bangkok |
| egypt | cairo |
| sweden | stockholm |
| indonesia | jakarta |
| poland | warsaw |
| chile | santiago |
| portugal | lisbon |
| iraq | baghdad |
| syria | damascus |
| peru | lima |
| guyana | georgetown |
| lebanon | beirut |
| hungary | budapest |
| kenya | nairobi |
| cuba | havana |
| norway | oslo |
| vietnam | hanoi |
| turkey | ankara |
| finland | helsinki |
| serbia | belgrade |
| pakistan | islamabad |
| bahamas | nassau |
| venezuela | caracas |
| libya | tripoli |
| bangladesh | dhaka |
| romania | bucharest |
| qatar | doha |
| nigeria | abuja |
| belarus | minsk |
| nepal | kathmandu |
| bosnia | sarajevo |
| ukraine | kyiv |
| ghana | accra |
| sudan | khartoum |
| azerbaijan | baku |
| senegal | dakar |
| latvia | riga |
| oman | muscat |
| croatia | zagreb |
| uganda | kampala |
| somalia | mogadishu |
| colombia | bogota |
| zimbabwe | harare |
| lithuania | vilnius |
| ecuador | quito |
| liberia | monrovia |
| iceland | reykjavik |
| slovakia | bratislava |

### (6) Common nouns

`common_nouns`: 384 items, 384 words.

| Category | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|---|---|---|---|---|
| animal | ant, egret, elephant, ferret, filly, gosling, hornet, husky, ibis, jellyfish, lizard, locust, newt, ocelot, pheasant, piglet | aardvark, bobcat, condor, deer, gander, gerbil, hamster, narwhal, oyster, retriever, rooster, rottweiler, spider, terrier, vole, weevil | barracuda, chickadee, corgi, coyote, crocodile, gator, iguana, jackal, macaque, manatee, mare, mule, racehorse, stork, tadpole, tortoise | dingo, dinosaur, dolphin, flamingo, hyena, koala, krill, llama, minnow, owl, penguin, puffin, rhino, snail, spaniel, stingray |
| device | accordion, brake, chainsaw, clarinet, gasket, headset, helm, laptop, lightbulb, magnet, mandolin, periscope, robot, saxophone, seatbelt, videotape | barometer, binder, buzzer, caliper, clipper, colander, computer, converter, crusher, laser, rudder, sprinkler, stapler, timer, typewriter, zipper | capacitor, compressor, dredge, keyboard, microscope, oboe, reactor, reflector, resistor, rotor, simulator, telescope, transistor, tripod, vise, xylophone | antenna, banjo, buckle, cymbal, dowel, dynamo, engine, jigsaw, paintbrush, sitar, spool, sundial, treadmill, trombone, winch, windmill |
| food | apple, biscuit, blackberry, brisket, donut, fillet, halibut, limpet, mullet, pear, penne, romaine, sardine, smelt, trout, yogurt | anchovy, baguette, grape, grouper, lobster, mackerel, meatloaf, mussel, pea, pepperoni, polenta, rhubarb, squid, steak, swordfish, waffle | carp, cheese, chipotle, choc, chorizo, cornbread, cupcake, ghee, hake, noodle, partridge, pork, potato, scone, shortbread, tomato | crayfish, dumpling, eel, granola, herring, pancake, pastrami, pita, radish, raisin, ravioli, salami, sirloin, strawberry, strudel, whitefish |
| person | activist, botanist, cadet, dentist, economist, evangelist, historian, knight, parent, pharmacist, podiatrist, president, scientist, servant, tourist, zoologist | blogger, designer, developer, elder, messenger, peeler, reader, reporter, rider, roofer, snapper, tiger, trapper, voter, weaver, welder | actor, bricklayer, collector, competitor, composer, conductor, critic, curator, dancer, editor, emperor, masher, navigator, treasurer, vicar, visitor | admiral, appraiser, astronomer, bailiff, banker, candidate, carpenter, cleric, dressmaker, fisher, fishmonger, geographer, magistrate, senator, teammate, translator |
| plant | beet, bergamot, carnation, edelweiss, elm, hazelnut, hibiscus, mesquite, nutmeg, phlox, pine, pinwheel, privet, sedge, tumbleweed, wheat | aster, baobab, buttercup, cactus, cardamom, clover, coriander, cucumber, grapefruit, hydrangea, lavender, lotus, mangrove, sagebrush, sunflower, turmeric | artichoke, banana, catnip, cedar, chive, cottonwood, garlic, gorse, licorice, mallee, medic, mustard, saffron, snapdragon, thistle, thyme | aspen, cumin, daffodil, ginkgo, kiwi, lupin, pumpkin, raspberry, sassafras, spinach, tuna, vanilla, verbena, wasabi, yucca, zucchini |
| structure | cabaret, chalet, convent, dugout, embankment, garage, jetty, monument, mosque, obelisk, parapet, skylight, tenement, tent, viaduct, winery | ballroom, belfry, cabin, chamber, condo, courtroom, dam, factory, kitchen, motel, nursery, palisade, pantry, refinery, smelter, theatre | boathouse, clubhouse, consulate, coop, cottage, courthouse, courtyard, cubicle, farmhouse, gate, gatehouse, guesthouse, manor, outhouse, statehouse, workhouse | auditorium, basilica, bathhouse, bullpen, chateau, fountain, hangar, kennel, pagoda, rectory, rink, rotunda, sauna, shrine, smokehouse, stockade |

### (7) Brands

`brands`: 192 items, 192 words.

| Industry | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|---|---|---|---|---|
| beverage | chivas, hennessy, illy, nespresso, pabst, pepsi, powerade, tropicana | absolut, coors, evian, heineken, lipton, molson, smirnoff, snapple | baileys, bollinger, budweiser, fanta, macallan, martell, modelo, perrier | bacardi, campari, carlsberg, gatorade, keurig, krug, starbucks, suntory |
| carmaker | byd, chevrolet, citroen, lada, opel, porsche, volkswagen, volvo | hino, hyundai, infiniti, mclaren, nio, nissan, renault, saab | chrysler, ferrari, isuzu, mahindra, maserati, maybach, mazda, vauxhall | acura, audi, bugatti, cadillac, daewoo, subaru, suzuki, toyota |
| electronics | electrolux, gopro, lenovo, motorola, oppo, polaroid, roku, sony | belkin, denon, dyson, epson, garmin, nikon, nokia, nvidia | anker, fitbit, fujifilm, logitech, miele, philips, razer, sandisk | asus, huawei, oneplus, panasonic, ricoh, samsung, wacom, xiaomi |
| internet and software | cloudflare, dropbox, netflix, netscape, paypal, shopify, skype, spotify | coinbase, expedia, google, icloud, linkedin, salesforce, tiktok, wikipedia | deezer, fiverr, gmail, kaspersky, myspace, napster, twitter, yandex | airbnb, autodesk, facebook, instagram, snapchat, wechat, whatsapp, youtube |
| retail chain | etsy, lidl, nordstrom, publix, sephora, taobao, waitrose, zappos | costco, debenhams, groupon, lululemon, menards, petsmart, rakuten, tesco | homebase, kmart, kroger, meijer, myer, radioshack, selfridges, wegmans | aldi, autozone, carrefour, gamestop, ulta, walgreens, walmart, wickes |
| video game | bioshock, capcom, metroid, roblox, spyro, wario, xbox, xenoblade | activision, digimon, konami, niantic, nintendo, pokemon, splatoon, tekken | atlus, fortnite, maxis, myst, rayman, skylanders, tamagotchi, undertale | atari, gamecube, sega, starcraft, warcraft, warframe, wii, zynga |

### (8) Pseudowords

`pseudowords`: 80 items, 80 words.

beack, blork, bludshu, breag, broalk, chosh, claislar, clelt, clolfusk, crild, darshur, derdeesk, dralnird, drerk, drith, droon, drup, drupseg, dusfeg, dusposk, fealvub, flalk, fleng, flibnoom, foogmes, fornoolt, frarlo, frord, gegrosk, gloa, glosdoo, gomp, graisk, groork, hoadtast, hoaldoap, jisbont, joanbick, jonvilt, joosk, keeth, kemdom, koolt, meskock, naisgeag, nedrish, nenshoab, paig, poompisk, pooth, reand, saibvoal, sheank, shesvuth, slaib, slibgeal, slish, sloappam, snoock, snuld, spoost, stast, swemp, swisfoab, swoald, swoamkal, swusk, temthe, thebchos, tisraig, trint, valnup, vemmea, vemtud, vesh, voog, voorkes, wapfoock, wirreg, woagdoot

## Class names

The 200 kept class names of `class_names.tsv`, by category.

| Category | Class names |
|---|---|
| animal | ant, badger, beagle, beaver, bee, bison, boar, cheetah, chihuahua, cockroach, cougar, coyote, cricket, dragonfly, drake, eel, ferret, flamingo, fly, goldfish, goose, gorilla, hamster, hare, hen, hippo, husky, impala, jaguar, jay, jellyfish, koala, leopard, lion, lynx, maltese, mantis, mink, ostrich, otter, peacock, pelican, pig, pug, quail, ram, robin, scorpion, skunk, slug, snail, starfish, sturgeon, tick, tiger, vulture, weasel, zebra |
| clothing | apron, bikini, bonnet, bra, diaper, gown, jean, mask, sock, suit, sweatshirt, wig |
| container | backpack, bathtub, bucket, cauldron, hamper, pitcher, vase, wallet |
| device | binoculars, buckle, candle, cannon, carousel, chainsaw, guillotine, hook, knot, laptop, lighter, microphone, muzzle, nail, printer, projectile, reel, revolver, rifle, screen, speaker, swing, syringe, torch, tripod |
| food | bagel, burrito, cup, dough, espresso, hay, pizza |
| instrument | accordion, banjo, bell, cello, drum, flute, sax, trumpet, violin |
| other | acorn, banana, broccoli, broom, bubble, cleaver, cliff, closet, coast, corn, cradle, crib, cucumber, desk, dishwasher, dome, fig, hammer, hatchet, holster, lakeside, lemon, maze, microwave, mower, mushroom, necklace, orange, parachute, pineapple, plow, pole, quill, shovel, skillet, spindle, stage, strawberry, stretcher, teddy, television, throne, toaster, umbrella, vacuum, valley, velvet, volcano, washer, wool |
| plant | daisy |
| structure | barn, castle, church, dam, dock, fountain, greenhouse, grille, library, lighthouse, monastery, mosque, palace, prison, restaurant, terrace |
| vehicle | barrow, jeep, lifeboat, limo, liner, pickup, schooner, semi, submarine, tandem, tank, taxi, tractor, wreck |

## Two-clue items

`two_clue_pilot.tsv` holds the pilot grids used during development and `two_clue_confirmatory.tsv` the confirmatory grids. Their cells and column names are hashed into the two-clue dataset fingerprint, which seeds the layout and the scrambles, so an edit to either file, a column name included, means rendering the two-clue stimuli again. In them the `feature` column holds the broad clue and `base` the narrow clue.
