import { OpenAIEmbeddings } from "@langchain/openai";
import dotenv from "dotenv";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import readline from "readline";

// Load environment variables
dotenv.config();

/**
 * Calculate cosine similarity between two vectors
 * Cosine similarity = (A · B) / (||A|| * ||B||)
 */
function cosineSimilarity(vectorA, vectorB) {
  if (vectorA.length !== vectorB.length) {
    throw new Error("Vectors must have the same dimensions");
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vectorA.length; i++) {
    dotProduct += vectorA[i] * vectorB[i];
    normA += vectorA[i] * vectorA[i];
    normB += vectorB[i] * vectorB[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function main() {
  console.log("🤖 JavaScript LangChain Agent Starting...\n");

  // Initialize embeddings model
  const embeddings = new OpenAIEmbeddings();

  // Create MemoryVectorStore instance
  const vectorStore = await MemoryVectorStore.fromTexts([], [], embeddings);

  // Check for GitHub token
  if (!process.env.GITHUB_TOKEN) {
    console.error("❌ Error: GITHUB_TOKEN not found in environment variables.");
    console.log("Please create a .env file with your GitHub token:");
    console.log("GITHUB_TOKEN=your-github-token-here");
    console.log("\nGet your token from: https://github.com/settings/tokens");
    console.log("Or use GitHub Models: https://github.com/marketplace/models");
    process.exit(1);
  }

  console.log("=== Embedding Inspector Lab ===\n");
  console.log("Generating embeddings for three sentences...\n");

  // Define the three test sentences
  const sentences = [
    "The canine barked loudly.",
    "The dog made a noise.",
    "The electron spins rapidly."
  ];

  // Add sentences to vector store with metadata
  await vectorStore.addDocuments(
    sentences.map((text, index) => ({
      pageContent: text,
      metadata: {
        createdAt: new Date().toISOString(),
        index,
      },
    }))
  );

  // Print confirmation message
  console.log(`✅ Successfully stored ${sentences.length} sentences in the vector store.`);

  // Print each sentence that was added
  sentences.forEach((sentence, index) => {
    console.log(`Sentence ${index + 1}: ${sentence}`);
  });

  // Show the distances between the embeddings
  console.log("\n=== Embedding Vectors ===\n");
  for (let i = 0; i < sentences.length; i++) {
    const current = i;
    const next = (i + 1) % sentences.length;
    const similarity = cosineSimilarity(
      vectorStore.getVector(current),
      vectorStore.getVector(next)
    );
    console.log(`Cosine similarity between Sentence ${current + 1} and Sentence ${next + 1}: ${similarity.toFixed(4)}`);
  }

  console.log("\n📊 Observations:");
  console.log("- Each embedding is just an array of floating-point numbers");
  console.log("- Sentences 1 and 2 (about dogs) will have similar values in many dimensions");
  console.log("- Sentence 3 (about electrons) will differ significantly from sentences 1 and 2");
  console.log("\nThis demonstrates that 'AI embeddings' are simply numerical vectors,");
  console.log("not magic—they represent semantic meaning as coordinates in high-dimensional space.");

  // Function to search sentences in the vector store
  async function searchSentences(vectorStore, query, k = 3) {
    // Perform similarity search
    const results = await vectorStore.similaritySearchWithScore(query, k);

    // Print the results
    console.log("\n=== Search Results ===\n");
    results.forEach(([document, score], index) => {
      console.log(
        `Rank ${index + 1}:\n` +
        `  Similarity Score: ${score.toFixed(4)}\n` +
        `  Sentence: ${document.text}\n`
      );
    });

    // Return the top k results
    return results;
  }

  // Example search
  console.log("\n=== Example Search ===\n");
  searchSentences(vectorStore, "The dog made a noise.");

  console.log("\n=== Semantic Search ===\n");

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const promptUser = async () => {
    rl.question("Enter a search query (or 'quit' to exit): ", async (input) => {
      const query = input.trim();

      if (query.toLowerCase() === "quit" || query.toLowerCase() === "exit") {
        console.log("Goodbye! 👋");
        rl.close();
        process.exit(0);
      } else if (!query) {
        console.log("Please enter a valid query.\n");
        return promptUser();
      }

      const results = await searchSentences(vectorStore, query);
      console.log("\n");
      return promptUser();
    });
  };

  await promptUser();
}

main().catch(console.error);