using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Connectors.InMemory;

var heroes = new List<Hero>
{
    new Hero
    {
        Id = 1,
        Name = "Superman",
        Description = "The Man of Steel, Superman is an alien from Krypton with superhuman strength, speed, flight, and near invincibility. A symbol of hope, he protects Earth with his moral compass and his commitment to justice.",
        // Powers = ["Superhuman strength", "Flight", "Heat vision", "Freezing breath", "Super speed", "Enhanced hearing", "Healing factor"]
    },
    new Hero
    {
        Id = 2,
        Name = "Batman",
        Description = "The Dark Knight of Gotham, Batman is a billionaire vigilante who uses his intellect, martial arts expertise, and advanced technology to fight crime. Driven by the loss of his parents, he embodies justice through fear and strategy. He is also a billionaire with lots of money.I ne",
        // Powers = ["Exceptional martial artist", "Combat strategy", "Stealth", "Detective skills", "Indomitable will", "Peak human physical condition"]
    },
    new Hero
    {
        Id = 3,
        Name = "Wonder Woman",
        Description = "An Amazonian warrior princess, Wonder Woman possesses superhuman strength, agility, and combat skills. Armed with the Lasso of Truth and her indomitable spirit, she fights for peace and equality as a champion of Themyscira.",
        // Powers = ["Superhuman strength", "Flight", "Combat skill", "Combat strategy", "Superhuman agility", "Healing factor", "Magic weaponry"]
    },
    new Hero
    {
        Id = 4,
        Name = "Flash",
        Description = "The Scarlet Speedster, Flash is a hero with the ability to move at incredible speeds, thanks to his connection to the Speed Force. Known for his quick wit and big heart, he races to save lives and outpace evil.",
        // Powers = ["Super speed", "Time travel", "Intangibility", "Superhuman agility", "Superhuman durability", "Superhuman stamina", "Electrokinesis"]
    }
};

var vectorStore = new InMemoryVectorStore();
var heroCollection = vectorStore.GetCollection<int, Hero>("heroes");
await heroCollection.CreateCollectionIfNotExistsAsync();

var generator = new OllamaEmbeddingGenerator(new Uri("http://localhost:11434"), "all-minilm:33m");

Console.WriteLine("Creating embeddings for heroes...");
foreach (var hero in heroes)
{
    Console.Write(".");
    hero.Embedding = await generator.GenerateEmbeddingVectorAsync(hero.Description);
    await heroCollection.UpsertAsync(hero);
}

Console.WriteLine("What's your emergency?");
var input = Console.ReadLine();

ArgumentNullException.ThrowIfNull(input);

Console.WriteLine("Searching for heroes...");
Console.WriteLine();

var queryEmbedding = await generator.GenerateEmbeddingVectorAsync(input);
var searchOptions = new VectorSearchOptions { Top = 2, VectorPropertyName = nameof(Hero.Embedding) };
var results = await heroCollection.VectorizedSearchAsync(queryEmbedding, searchOptions);

await foreach (var result in results.Results)
{
    Console.WriteLine($"Name: {result.Record.Name}");
    Console.WriteLine($"Description: {result.Record.Description}");
    // Console.WriteLine($"Powers: {string.Join(", ", result.Record.Powers)}");
    Console.WriteLine($"Similarity: {result.Score}");
    Console.WriteLine();
}

public class Hero
{
    [VectorStoreRecordKey]
    public int Id { get; set; }

    [VectorStoreRecordData]
    public string Name { get; set; }

    [VectorStoreRecordData]
    public string Description { get; set; }

    // [VectorStoreRecordData]
    // public string[] Powers { get; set; }

    [VectorStoreRecordVector(1600, DistanceFunction.CosineSimilarity)]
    public ReadOnlyMemory<float> Embedding { get; set; }
}
