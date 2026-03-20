# This is a simple example
import asyncio

from mellea.backends import ModelOption
from mellea_ibm.rits import RITSBackend, RITS

# Local imports
from fact_reasoner.core.nli import NLIExtractor

# Create a Mellea RITS backend
backend = RITSBackend(
    RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 4096}
)

# Create the NLI extractor
extractor = NLIExtractor(backend)    

# Example premise and hypothesis
premise = "natural born killers is a 1994 american romantic crime action film directed by oliver stone and starring woody harrelson, juliette lewis, robert downey jr., tommy lee jones, and tom sizemore. the film tells the story of two victims of traumatic childhoods who become lovers and mass murderers, and are irresponsibly glorified by the mass media. the film is based on an original screenplay by quentin tarantino that was heavily revised by stone, writer david veloz, and associate producer richard rutowski. tarantino received a story credit though he subsequently disowned the film. jane hamsher, don murphy, and clayton townsend produced the film, with arnon milchan, thom mount, and stone as executive producers. natural born killers was released on august 26, 1994 in the united states, and screened at the venice film festival on august 29, 1994. it was a box office success, grossing $ 110 million against a production budget of $ 34 million, but received polarized reviews. some critics praised the plot, acting, humor, and combination of action and romance, while others found the film overly violent and graphic. notorious for its violent content and inspiring \" copycat \" crimes, the film was named the eighth most controversial film in history by entertainment weekly in 2006. = = plot = = mickey knox and his wife mallory stop at a diner in the new mexico desert. a duo of rednecks arrive and begin sexually harassing mallory as she dances by a jukebox. she initially encourages it before beating one of the men viciously. mickey joins her, and the couple murder everyone in the diner, save one customer, to whom they proudly declare their names before leaving. the couple camp in the desert, and mallory reminisces about how she met mickey, a meat deliveryman who serviced her family ' s household. after a whirlwind romance, mickey is arrested for grand theft auto and sent to prison ; he escapes and returns to mallory ' s home. the couple murders mallory ' s sexually abusive father and neglectful mother, but spare the life of mallory ' s little brother, kevin. the couple then have an unofficial marriage ceremony on a bridge. later, mickey and mallory hold a woman hostage in their hotel room. angered by mickey ' s desire for a threesome, mallory leaves, and mickey rapes the hostage. mallory drives to a nearby gas station, where she flirts with a mechanic. they begin to have sex on the hood of a car, but after mallory suffers a flashback of being raped by her father, and the mechanic recognizes her as a wanted murderer, mallory kills him. the pair continue their killing spree, ultimately claiming 52 victims in new mexico, arizona and nevada. pursuing them is detective jack scagnetti, who became obsessed with mass murderers at the age of eight after having witnessed the murder of his mother at the hand of charles whitman. beneath his heroic facade, he is also a violent psychopath and has murdered prostitutes in his past. following the pair ' s murder spree is self - serving tabloid journalist wayne gale, who profiles them on his show american maniacs, soon elevating them to cult - hero status. mickey and mallory become lost in the desert after taking psychedelic mushrooms, and they stumble upon a ranch owned by warren red cloud, a navajo man who provides them food and shelter. as mickey and mallory sleep, warren, sensing evil in the couple, attempts to exorcise the demon that he perceives in mickey, chanting over him as he sleeps. mickey, who has nightmares of his abusive parents, awakens during the exorcism and shoots warren to death. as the couple flee, they feel inexplicably guilty and come across a giant field of rattlesnakes, where they are badly bitten. they reach a drugstore to purchase snakebite antidote, but the store is sold out. a pharmacist recognizes the couple and triggers an alarm before mickey kills him. police arrive shortly after and accost the couple and a shootout ensues. the police end the showdown by beating the couple while a "
hypothesis = "Lanny Flaherty has appeared in numerous films."

# Run the NLI extractor
result = extractor.run(premise=premise, hypothesis=hypothesis)
print(f"H -> P: {result}")

# More examples of premises and hyptheses
premises = [
    "The biggest risk facing the world's insurance companies is possibly the rapid change now taking place within their own ranks. Sluggish growth in core markets and intense price competition, coupled with shifting patterns of customer demand and the rising cost of losses, are threatening to overwhelm those too slow to react.",
    "The biggest risk facing the world's insurance companies is possibly the rapid change now taking place within their own ranks. Sluggish growth in core markets and intense price competition, coupled with shifting patterns of customer demand and the rising cost of losses, are threatening to overwhelm those too slow to react.",
    "The biggest risk facing the world's insurance companies is possibly the rapid change now taking place within their own ranks. Sluggish growth in core markets and intense price competition, coupled with shifting patterns of customer demand and the rising cost of losses, are threatening to overwhelm those too slow to react.",
]
hypotheses = [
    "Insureance companies are experiencing a boom in their core markets.",
    "Insureance companies are competing to provide the best service to their customers.",
    "Customers don't trust insurance companies as much as they once were.",
]

# Run the batch of premises and hypotheses
result = asyncio.run(extractor.run_batch(premises=premises, hypotheses=hypotheses))

# Output results
print(f"H -> P: {result}")
print("Done.")