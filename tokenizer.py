#!/usr/bin/env python
# -*- coding: utf-8 -*

import nltk
from collections import defaultdict
import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


text = """President Obama on Tuesday issued his strongest warning yet about the Republican candidates for president with a two-word message on why voters should choose solemnly: nuclear codes.

Bluntly questioning front-runner Donald Trump's temperament, Obama said, "Whoever is standing where I'm standing right now has the nuclear codes with them and can order 21-year-olds into a firefight."

A restive electorate ultimately will decline to elect Trump, he predicted.

"The American people are pretty sensible," Obama said. "And I think they'll make a sensible choice in the end."

Though he referred specifically to Trump, Obama also took care to warn about all of the GOP candidates.

"Not a single one of them" is talking about some of the world's biggest problems, he said.

The words represented Obama's most energetic criticism of the Republicans running to replace him. For months, he has mostly kept a studied distance from the fray and resisted invitations to engage in political analysis.

But GOP candidates are promising to dismantle Obama's entire legacy if they win the Oval Office, and polls show Trump dramatically in the lead in South Carolina going into its Saturday primary. Such numbers are infusing his campaign with more momentum on the heels of his decisive win in New Hampshire last week.

Asked about the president's comments at a South Carolina forum, Trump responded with relative restraint, saying only that Obama had done a "lousy job as president" and that he would have defeated him in 2012 had he run.

"For him to say that is actually a great compliment," Trump argued of Obama's criticism.

As the president launches his agenda for his final year in office, aides say, Obama has been increasingly concerned about protecting his legacy, including his healthcare reform and immigration policy as well as his attempts to orient U.S. foreign policy away from war and toward diplomacy.

His comments Tuesday also reflected the view of someone who's long been critical of the hyperactive political environment — he at one point admonished not only politicians engaging in theater but also reporters who cover the campaign as "entertainment" — and whose outlook has been tempered by the responsibilities of the office for seven years.

Or, as Obama put it, he's someone who has "been a candidate of hope and change and a president who's got some nicks and cuts and bruises from, you know, getting stuff done over the last seven years."

After two days of meetings with 10 Southeast Asian leaders here at the presidential getaway estate of Sunnylands, Obama said he was also worried about what the campaign speeches and interviews were doing to American relations abroad.

Foreign observers are troubled by some of the rhetoric, he said. In the past, Obama had singled out Trump's pledge to ban Muslims from entering the country and deporting anyone not living in the country legally.

GOP brawl in South Carolina may have repercussions in the general election
GOP brawl in South Carolina may have repercussions in the general election
In a conversation with reporters at the close of the summit, Obama said Trump wasn't the only one he was worried about.

"This is not just Mr. Trump," Obama said. "There's not a single candidate in the Republican primary that thinks we should do anything about climate change.... The rest of the world looks at that and says, 'How can that be?'"

Voters are venting, he said, but ultimately "reality has a way of intruding."

"I have a lot of faith in the American people. And I think they recognize that being president is a serious job," he said. "It's not hosting a talk show or a reality show. It's not promotion. It's not marketing. It's hard. And a lot of people count on us getting it right."

Obama also downplayed the fight on the Democratic side between Hillary Clinton and Bernie Sanders, saying there was broad agreement in his party on principles but "a difference in tactics, trying to figure out how do you actually get things done."

Obama said he might eventually express his view in the race but that "for now I think it's important for Democratic voters to express themselves and for the candidates to be run through the paces."

"The thing I can say unequivocally," he said, "is I am not unhappy that I'm not on the ballot."

"""

text = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec cursus tempor enim, congue efficitur metus dictum eu. Quisque vulputate mi eu nisl efficitur, eu gravida ex facilisis. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec quis aliquam lorem. Nullam tortor arcu, malesuada a pellentesque eu, aliquet non diam. Nam lacinia ultricies augue, eu mattis tellus posuere quis. Donec varius aliquam dolor, a condimentum lorem consequat nec. Nam dictum bibendum lacinia. Integer nec diam vitae lacus placerat varius eget eu libero. Praesent at orci dignissim, porttitor turpis sit amet, iaculis felis. Sed tincidunt nulla ipsum, ut vulputate eros auctor tempus. Pellentesque blandit et turpis et fermentum. Integer a vestibulum massa. Etiam ac nisl ipsum. Curabitur ullamcorper rutrum tristique.

Vestibulum magna sapien, scelerisque in convallis eu, vulputate eget velit. Aenean quis pretium leo. Morbi diam nisi, lobortis porta dignissim eget, vehicula at ante. Praesent ante est, consequat vel lectus a, pulvinar viverra justo. Mauris sagittis, purus vel hendrerit faucibus, augue dui venenatis orci, vel mollis elit eros quis magna. Maecenas consectetur quam condimentum lorem suscipit, non lobortis augue tincidunt. Quisque id consectetur nulla, sed gravida sem. Aenean vitae eros vel diam cursus faucibus. Vestibulum ut dolor vehicula, ullamcorper est in, tristique ligula.

Suspendisse tincidunt iaculis dolor id mollis. In hac habitasse platea dictumst. Vestibulum ut dignissim ipsum, dapibus pulvinar erat. Praesent consequat sagittis tincidunt. Duis euismod sollicitudin magna vel laoreet. Suspendisse ut elementum arcu, ac dignissim nisl. Fusce finibus mollis odio sed mattis. Phasellus dictum arcu sed rhoncus sodales. Fusce dapibus metus at purus scelerisque condimentum. Aliquam posuere dapibus dapibus. Phasellus auctor turpis mi, non viverra turpis aliquam et. Vivamus eget risus ac felis euismod interdum. Praesent tristique, lorem sit amet consectetur accumsan, urna elit euismod eros, sed maximus lorem felis non velit. Quisque a fermentum lectus.

Vestibulum fringilla maximus augue quis dictum. Suspendisse quis tempus ipsum. Duis ut commodo lorem, ut auctor urna. Sed eleifend dolor eu neque pretium rutrum. Praesent nulla elit, fermentum non ex vitae, imperdiet blandit turpis. Maecenas in laoreet augue, vel congue libero. Donec luctus molestie porttitor. Ut at ultricies leo. Nulla consectetur elementum felis, tincidunt placerat mi dapibus eu. Fusce vitae sem nec purus laoreet lobortis non id erat. Donec efficitur semper molestie. Pellentesque iaculis massa dapibus, sollicitudin sem quis, efficitur est. Suspendisse in libero posuere, dapibus purus porttitor, volutpat ligula. Ut velit leo, maximus et eros euismod, rhoncus rhoncus ex. Quisque orci neque, tempus sit amet lacus a, feugiat pharetra nisl.

Mauris vitae mollis turpis. Sed eu nisi nec purus accumsan eleifend ut nec risus. Nunc ac tellus lobortis, imperdiet sem id, ultricies odio. Nunc iaculis nisl a arcu ornare, vitae venenatis est vehicula. Integer venenatis dolor egestas, pretium est ut, malesuada diam. Phasellus bibendum feugiat laoreet. Donec condimentum consequat nulla, condimentum pretium leo fermentum ac. Aliquam varius tempor risus, non lobortis nibh consequat vel. Curabitur tempus vestibulum aliquet. Donec finibus magna quis elit iaculis fringilla. Donec viverra elit eu sapien venenatis, quis vulputate diam dignissim. Vestibulum ex nulla, mollis et pellentesque in, vestibulum at tellus. Praesent et enim gravida, vehicula est quis, gravida risus. In a quam nibh.

Quisque vehicula elit at turpis dignissim venenatis id vel ante. Sed pulvinar sit amet metus et facilisis. Aliquam erat volutpat. Duis tristique lorem enim, in ultrices arcu mattis eu. Vivamus et consectetur ligula, at consequat ipsum. Donec ex quam, imperdiet et bibendum non, mollis a mauris. Phasellus tincidunt porttitor vestibulum. Donec aliquam vehicula ante, nec elementum tortor gravida non.

Pellentesque aliquet vestibulum augue a suscipit. Mauris venenatis volutpat lorem, sit amet tempus nisi ornare eu. Duis id ipsum sed felis gravida vulputate at vel eros. Ut aliquam lorem mauris, tincidunt viverra tortor blandit ac. Suspendisse pulvinar fermentum ullamcorper. Nullam risus ex, porttitor id eros sit amet, pharetra congue augue. Fusce ut euismod sapien. Vivamus laoreet nulla nec lacus luctus, sed dictum ante tincidunt. Suspendisse potenti. Nullam vel nunc nunc. Curabitur sed vestibulum metus. Vivamus dictum risus quis ligula auctor, sollicitudin finibus elit tincidunt. Vivamus condimentum nisl eget lectus bibendum, sit amet molestie quam vestibulum. Vivamus cursus eu odio a accumsan. Morbi nisi diam, molestie vitae augue nec, tincidunt posuere odio. Nulla suscipit est vitae velit cursus pulvinar.

Maecenas elementum turpis ut mauris sollicitudin sodales. Sed nec risus volutpat dolor volutpat dapibus. In sed sodales ante, quis elementum nibh. Praesent congue, est in interdum imperdiet, lorem sapien vulputate justo, non lacinia quam ligula eu sapien. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc a hendrerit ipsum. Vestibulum elit magna, sodales sit amet turpis eget, lobortis ultricies est. Cras justo lectus, bibendum quis sodales et, malesuada non ipsum. Mauris id augue ac lorem gravida congue eu vitae sapien.

Nam non ante odio. Vivamus eu sapien tortor. Morbi pulvinar tellus eget mi iaculis ullamcorper. Aenean velit eros, condimentum ac mollis sit amet, imperdiet nec nisi. In quis feugiat tellus. Curabitur vulputate felis orci, ornare tristique nisl efficitur a. Aenean tempor aliquam pretium.

Sed sapien nunc, sodales in velit sit amet, scelerisque pharetra lectus. Curabitur sed mauris sapien. Vivamus sit amet tempor orci. Duis iaculis nulla lectus, in luctus purus ullamcorper quis. In ut tempus est, nec aliquet neque. Nullam eget diam orci. Sed a luctus mi. Vestibulum sit amet erat arcu. Mauris in neque in nisl sollicitudin cursus sit amet nec dolor. Integer mi odio, imperdiet ac neque blandit, viverra lobortis neque. Fusce rhoncus, ex eu volutpat sagittis, magna lacus laoreet felis, sed mollis elit orci blandit turpis. Sed facilisis orci quis velit congue, non tincidunt diam pulvinar. Fusce ut porttitor lacus. Phasellus bibendum, nunc sed mollis imperdiet, ipsum lacus aliquet urna, ornare dignissim erat lacus hendrerit nulla. Quisque dolor urna, pellentesque nec tincidunt quis, condimentum ut est. Vestibulum a justo nec tortor tempus dictum in a purus.
"""

words = nltk.word_tokenize(text)
bigrams = nltk.bigrams(words)
trigrams = nltk.trigrams(words)

print text
print '\n'

print words
print '\n'

d = defaultdict(int)
for x in bigrams:
	d[x] += 1
print sorted(d.items(), key=lambda x: x[1], reverse=True)

print '\n'

d = defaultdict(int)
for x in trigrams:
	d[x] += 1
print sorted(d.items(), key=lambda x: x[1], reverse=True)
print '\n'

# vect = CountVectorizer(ngram_range=(1,10))
# analyzer = vect.build_analyzer()
# print analyzer(text)

# text = text.encode('ascii', 'ignore').decode('utf-8', 'ignore')

text = 'hello world my name is davis \n da beast monster man dude davis davis hello'
text = text.decode(errors='replace').split('\n')

vect = TfidfVectorizer(
	encoding='utf-8',
	ngram_range=(1, 5),
	decode_error='ignore',
	analyzer='word',
	stop_words='english'
)
X = vect.fit_transform(text)
print X
idf = vect.idf_
d = dict(zip(vect.get_feature_names(), idf))
pprint.pprint(d)
