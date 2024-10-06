import { StreamingTextResponse, LangChainStream, Message } from 'ai';
import { ChatOpenAI } from 'langchain/chat_models/openai';

import { ConversationalRetrievalQAChain } from 'langchain/chains';
import { vectorStore } from '@/utils/openai';
import { NextResponse } from 'next/server';
import { BufferMemory } from "langchain/memory";
import { PineconeClient } from '@pinecone-database/pinecone';

const pinecone = new PineconeClient();
await pinecone.init({
  apiKey: process.env.PINECONE_API_KEY,
  environment: process.env.PINECONE_ENVIRONMENT,
});

const index = pinecone.Index('your-index-name');

// Function to upsert embeddings
async function upsertEmbeddings(texts, vectors) {
  await index.upsert({
    vectors: vectors.map((vector, idx) => ({
      id: `text-${idx}`,
      values: vector,
      metadata: { text: texts[idx] },
    })),
  });
}

// Function to query embeddings
async function queryEmbeddings(vector) {
  const queryResponse = await index.query({
    queryVector: vector,
    topK: 5,
    includeMetadata: true,
  });
  return queryResponse.matches;
}

export async function POST(req: Request) {
    try {
        const { stream, handlers } = LangChainStream();
        const body = await req.json();
        const messages: Message[] = body.messages ?? [];
        const question = messages[messages.length - 1].content;

        const model = new ChatOpenAI({
            temperature: 0.8,
            streaming: true,
            callbacks: [handlers],
        });

        const retriever = vectorStore().asRetriever({ 
            "searchType": "mmr", 
            "searchKwargs": { "fetchK": 10, "lambda": 0.25 } 
        })
        const conversationChain = ConversationalRetrievalQAChain.fromLLM(model, retriever, {
            memory: new BufferMemory({
              memoryKey: "chat_history",
            }),
          })
        conversationChain.invoke({
            "question": question
        })

        return new StreamingTextResponse(stream);
    }
    catch (e) {
        return NextResponse.json({ message: 'Error Processing' }, { status: 500 });
    }
}