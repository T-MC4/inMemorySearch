// Import the required libraries
import * as tf from '@tensorflow/tfjs'; // Need this of tfjs-node
import * as tfn from '@tensorflow/tfjs-node'; // Need this to make queries 100x faster
import * as encoder from '@tensorflow-models/universal-sentence-encoder';
import '@tensorflow/tfjs-backend-cpu'; // Import the CPU and WebGL backends to increase performance
import pkg from 'hnswlib-node'; // Import the HNSW library
const { HierarchicalNSW } = pkg;
import path from 'path'; // Import the file system library
import fs from 'fs/promises';
import fillerMap from './fillerMap.js';
import {
    getFilesInDirectory,
    readFileContent,
} from './customTextAndJSONLoaders.js'; // Import the custom text and JSON loaders

/**
 * Load the model from the tensorflow hub.
 * @param {boolean} debug - Whether to print debug information
 *
 * @returns - The model
 */
async function loadModel(debug = false) {
    const start = performance.now();

    const model = await encoder.load();

    if (debug) {
        console.log(
            `\nModel Loading took ${performance.now() - start} milliseconds.`
        );
    }
    return model;
}

/**
 * Convert the text to an embedding.
 *
 * @param {Object} model - The model to use for the conversion
 * @param {array} texts - The texts to convert
 * @param {boolean} debug - Whether to print debug information
 * @returns - The embedding - a 2D array
 */
async function convertToEmbedding(model, texts, debug = false) {
    const start = performance.now();

    const embeddings = await model.embed(texts);
    const embeddingArray = await embeddings.array();

    embeddings.dispose(); // Clean up the memory

    if (debug) {
        console.log(
            `\nEmbedding took ${
                performance.now() - start
            } milliseconds. shape ${embeddingArray.length}`
        );
    }
    return embeddingArray;
}

/**
 * Search the sentence in the indexing and return the nearest neighbors.
 *
 * @param {string} sentences - The sentence to search in a list of sentences
 * @param {Object} model - The model to use for the conversion
 * @param {Object} indexing - The indexing to use for the search
 * @param {number} nearestNeighbors - The number of nearest neighbors to return
 * @param {boolean} debug - Whether to print debug information
 * @returns - The nearest neighbors which contains distances and IDs.
 */
async function vectorSearch(
    sentences,
    model,
    indexing,
    nearestNeighbors,
    debug = false
) {
    // Convert the sentence to an embedding.
    const queryVector = await convertToEmbedding(model, sentences, debug);

    const start = performance.now();

    const result = indexing.searchKnn(queryVector[0], nearestNeighbors);

    if (debug) {
        console.log(`\nSearch took ${performance.now() - start} milliseconds.`);
    }

    return result;
}

/**
 * Build the indexing.
 *
 * @param {number} numDimensions - The number of dimensions
 * @param {number} maxElements - The maximum number of elements
 * @param {array} embeddings - The embeddings
 * @param {array} IDs - The IDs
 * @param {boolean} debug - Whether to print debug information
 * @returns - The indexing
 */
function buildIndexing(
    numDimensions,
    maxElements,
    embeddings,
    IDs,
    debug = false
) {
    const start = performance.now();

    const indexing = new HierarchicalNSW('l2', numDimensions);
    indexing.initIndex(maxElements);
    embeddings.forEach((embedding, index) => {
        indexing.addPoint(embedding, IDs[index]);
    });

    if (debug) {
        console.log(
            `\nBuilding Index took ${performance.now() - start} milliseconds.`
        );
    }
    return indexing;
}

/**
 * Create an ID for each page.
 *
 * @param {number} index - The index of the page
 * @param {number} fillerID - The filler ID
 * @returns - The ID
 */
function createID(index, fillerID) {
    const id = index * 100 + fillerID;
    return id;
}

/**
 * Get the filler ID from the ID.
 *
 * @param {number} id - The ID
 * @returns - The filler ID
 */
function getFillerID(id) {
    return id % 100;
}

/**
 * Read all the files in the directory and return the content.
 *
 * @param {string} baseDir - The base directory
 * @param {string} extension - The extension of the files to read.
 * @param {boolean} debug - Whether to print debug information
 * @returns - Array which contains the page content and the ID.
 */
async function readAllFilesContent(baseDir, extension, debug = false) {
    var start = performance.now();
    // Load data to build the indexing
    const arrayOfJSONFiles = await getFilesInDirectory(baseDir, extension);

    var jsonContent = [];

    // Process all the new files
    for (const fileName of arrayOfJSONFiles) {
        // Read the contents
        const filePath = path.join(baseDir, fileName);
        const content = await readFileContent(filePath);
        jsonContent = [...jsonContent, ...JSON.parse(content)];

        // Move file to 'processed' folder indicate it's done
        const destPath = `./data/processed/${fileName}`;
        await fs.rename(filePath, destPath);
        console.log('File Read & Moved successfully');
    }

    // Load the existing index
    const contents = [];
    const ids = [];

    // Update the existing index
    jsonContent.forEach((entry, index) => {
        contents.push(entry.pageContent);
        ids.push(createID(index, entry.metadata.fillerID));
    });

    if (debug) {
        console.log(
            `\nReading all files took ${
                performance.now() - start
            } milliseconds.`
        );
    }
    return {
        contents,
        ids,
    };
}

async function main() {
    const DEBUG = true;
    // Load data to build the indexing

    const numDimensions = 512; // the length of data point vector that will be indexed.
    const maxElements = 10000; // the maximum number of data points.
    const nearestNeighbors = 3;

    const model = await loadModel(DEBUG);
    const { ids, contents } = await readAllFilesContent(
        './data/to_process',
        'json',
        DEBUG
    );
    const embeddings = await convertToEmbedding(model, contents, DEBUG);

    const indexing = buildIndexing(
        numDimensions,
        maxElements,
        embeddings,
        ids,
        DEBUG
    );

    // Perform a nearest neighbor search
    const sentence = [
        `I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary
		is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following
		CPU instructions in performance-critical operations:  AVX2 FMA
      I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary
      is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following
      CPU instructions in performance-critical operations:  AVX2 FMAoperations:  AVX2 FMA`,
        // 'Yeah.',
    ];

    // Push top_k fillerText values here
    const matches = [];

    for (let text of sentence) {
        // Perform the vector search
        let result = await vectorSearch(
            [text],
            model,
            indexing,
            nearestNeighbors,
            DEBUG
        );

        // Get the fillerText associated with the search result
        let start = performance.now();
        const fillers = result.neighbors.map((id) => {
            return fillerMap.get(getFillerID(id));
        });
        const resultWithFiller = { ...result, fillers };
        matches.push(resultWithFiller);

        // Log the performance
        console.table(result);
        if (DEBUG) {
            console.log(
                `\nSearching post processing took ${
                    performance.now() - start
                } milliseconds (ie. converting embedding ID into fillerText value).`
            );
        }
    }
    return matches;
}

const [match] = await main();
console.log(match);
