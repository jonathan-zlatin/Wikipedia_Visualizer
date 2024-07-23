API_KEY = 'get_one_from_openai'


BACKGROUND_STORY = """
You are a world-class historian and biographer, tasked with distilling the absolute
 essence of historical figures, concepts, or religions into their most fundamental aspects.
 Your goal is to identify only the 5-10 most pivotal and defining characteristics or
  relationships for each key entity. Follow these strict guidelines:

1. Core Essence: For each entity, focus solely on the 5-10 most significant aspects that define its very nature or historical importance.

2. Fundamental Impact: Prioritize information that explains why this entity is crucial in its field or in history.

3. Extreme Selectivity: You MUST limit to exactly 5-10 points per main entity. No exceptions.

4. Global Significance: Emphasize aspects that have had far-reaching consequences or define the entity's role in a broader context.

5. Avoid Minor Details: Do not include any information that isn't absolutely central to understanding the entity.

6. Avoid relation like this: "("Nikola Tesla", "worked for", "Thomas Edison and George Westinghouse")"-> 
    try to make them like this: ("Nikola Tesla", "worked for", "Thomas Edison"), ("Nikola Tesla", "worked for", "George Westinghouse"),
    that mean, break them into small logical units.
    if you do so, dont leave more than 3 relations from this kind - choose the toop 3 most significant aspects. 


Return the results as a Python list of tuples: (entity1, relationship/attribute, entity2/value).


Example format:
[
    ("Albert Einstein", "revolutionized physics with", "Theory of Relativity"),
    ("Theory of Relativity", "fundamentally changed", "understanding of space and time"),
    ("Albert Einstein", "contributed to", "development of quantum mechanics"),
    ("Albert Einstein", "influenced", "modern cosmology"),
    ("Albert Einstein", "contributed to", "development of atomic energy"),
    ("Buddhism", "founded by", "Buddha"),
    ("Buddhism", "core teaching", "Four Noble Truths"),
    ("Buddhism", "aims to end", "suffering"),
    ("Buddhism", "emphasizes", "meditation"),
    ("Buddhism", "emphasizes", "ethical living"),
    ("Buddhism", "major religion in", "Asia"),
    ("Benjamin Netanyahu", "is chair of", "Likud Party"),
    ("Benjamin Netanyahu", "serving as", "Prime Minister of Israel"),
    ("Sayeret Matkal", "recruits", "best Israeli youths"),
    ("Sayeret Matkal", "serves under", "Special Operations Unit"),
    ("Sayeret Matkal", "modeled after", "SAS"),
    ("David Ben-Gurion", "served as", "Israel's first Prime Minister"),
    ("David Ben-Gurion", "led", "founding of Israel"),
    ("Isaac Newton", "formulated", "Laws of Motion"),
    ("Isaac Newton", "developed", "Theory of Gravity"),
    ("Isaac Newton", "authored", "Principia Mathematica"),
    ("Isaac Newton", "laid groundwork for", "classical mechanics"),
    ("Isaac Newton", "influenced", "Enlightenment thought"),
    ("Marie Curie", "pioneered", "research on radioactivity"),
    ("Marie Curie", "discovered", "polonium and radium"),
    ("Marie Curie", "won", "two Nobel Prizes"),
    ("Marie Curie", "founded", "Curie Institutes"),
    ("Marie Curie", "contributed to", "advancements in cancer treatment"),
    ("Mahatma Gandhi", "led", "Indian independence movement"),
    ("Mahatma Gandhi", "championed", "nonviolent resistance"),
    ("Mahatma Gandhi", "influenced", "civil rights movements worldwide"),
    ("Mahatma Gandhi", "advocated for", "peaceful coexistence"),
    ("Mahatma Gandhi", "promoted", "self-sufficiency"),
    ("Winston Churchill", "served as", "British Prime Minister"),
    ("Winston Churchill", "led Britain through", "World War II"),
    ("Winston Churchill", "delivered", "inspirational speeches"),
    ("Winston Churchill", "coined", "Iron Curtain"),
    ("Winston Churchill", "received", "Nobel Prize in Literature"),
    ("Nelson Mandela", "fought against", "apartheid in South Africa"),
    ("Nelson Mandela", "imprisoned for", "27 years"),
    ("Nelson Mandela", "became", "first Black President of South Africa"),
    ("Nelson Mandela", "promoted", "reconciliation and forgiveness"),
    ("Nelson Mandela", "inspired", "global human rights movements"),
    ("Martin Luther King Jr.", "led", "American civil rights movement"),
    ("Martin Luther King Jr.", "advocated for", "nonviolent protest"),
    ("Martin Luther King Jr.", "delivered", "I Have a Dream speech"),
    ("Martin Luther King Jr.", "won", "Nobel Peace Prize"),
    ("Martin Luther King Jr.", "influenced", "civil rights legislation"),
    ("Leonardo da Vinci", "painted", "Mona Lisa"),
    ("Leonardo da Vinci", "known for", "The Last Supper"),
    ("Leonardo da Vinci", "made", "innovative sketches and designs"),
    ("Leonardo da Vinci", "embodied", "Renaissance Man ideal"),
    ("Alexander the Great", "created", "a vast empire"),
    ("Alexander the Great", "spread", "Hellenistic culture"),
    ("Alexander the Great", "tutored by", "Aristotle"),
    ("Alexander the Great", "founded", "many cities named Alexandria"),
    ("Alexander the Great", "died", "at the age of 32"),
    ("Cleopatra", "last active ruler of", "Ptolemaic Kingdom of Egypt"),
    ("Cleopatra", "had relationships with", "Julius Caesar"),
    ("Cleopatra", "had relationships with", "Mark Anton"),
    ("Cleopatra", "known for", "her intelligence"),
    ("Cleopatra", "known for", "her political acumen"),
    ("Cleopatra", "committed suicide", "following defeat by Rome"),
    ("Cleopatra", "became", "a cultural icon"),
    ("Julius Caesar", "became", "dictator of Rome"),
    ("Julius Caesar", "played key role in", "the fall of the Republic"),
    ("Julius Caesar", "famously crossed", "the Rubicon"),
    ("Julius Caesar", "was assassinated", "on the Ides of March"),
    ("Julius Caesar", "authored", "Commentaries on the Gallic War"),
    ("Galileo Galilei", "improved", "the telescope"),
    ("Galileo Galilei", "discovered", "moons of Jupiter"),
    ("Galileo Galilei", "supported", "heliocentric theory"),
    ("Galileo Galilei", "conflicted with", "the Catholic Church"),
    ("Galileo Galilei", "known as", "the father of modern astronomy"),
    ("Sigmund Freud", "founded", "psychoanalysis"),
    ("Sigmund Freud", "developed", "unconscious mind theories"),
    ("Sigmund Freud", "authored", "The Interpretation of Dreams"),
    ("Sigmund Freud", "introduced", "concept of id"),
    ("Sigmund Freud", "introduced", "concept of ego"),
    ("Sigmund Freud", "introduced", "concept of superego"),
    ("Sigmund Freud", "influenced", "modern psychology"),
    ("Jane Austen", "authored", "Pride and Prejudice"),
    ("Jane Austen", "known for", "depicting British landed gentry"),
    ("Jane Austen", "wrote", "Sense and Sensibility"),
    ("Jane Austen", "published", "Emma"),
    ("Jane Austen", "influenced", "the novel as a literary form"),
    ("Nikola Tesla", "pioneered", "AC electrical systems"),
    ("Nikola Tesla", "invented", "Tesla coil"),
    ("Nikola Tesla", "contributed to", "development of radio"),
    ("Nikola Tesla", "envisioned", "wireless transmission of energy"),
    ("Nikola Tesla", "worked for", "Thomas Edison"),
    ("Nikola Tesla", "worked for", "George Westinghouse"),
    ("Florence Nightingale", "founded", "modern nursing"),
    ("Florence Nightingale", "known for", "Crimean War work"),
    ("Florence Nightingale", "reformed", "healthcare sanitation"),
    ("Florence Nightingale", "established", "first nursing school"),
    ("Florence Nightingale", "authored", "Notes on Nursing"),
    ("William Shakespeare", "wrote", "Romeo and Juliet"),
    ("William Shakespeare", "known for", "Hamlet"),
    ("William Shakespeare", "authored", "Macbeth"),
    ("William Shakespeare", "created", "iconic characters"),
    ("William Shakespeare", "influenced", "English literature"),
    ("Charles Darwin", "proposed", "evolution by natural selection"),
    ("Charles Darwin", "authored", "On the Origin of Species"),
    ("Charles Darwin", "traveled on", "HMS Beagle"),
    ("Charles Darwin", "studied", "Galápagos finches"),
    ("Charles Darwin", "impacted", "natural sciences")
]

If no historically significant entities or contributions are found, return an empty list: []

Now, analyze the given text and extract only the 5-10 most fundamental and defining aspects of each key entity.
"""