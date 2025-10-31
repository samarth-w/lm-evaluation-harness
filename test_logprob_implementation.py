#!/usr/bin/env python3
"""
Test script to validate different OpenVINO GenAI logprob approaches
"""

import logging
import openvino_genai as ov_genai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_different_approaches():
    """Test different ways to get logprobs from OpenVINO GenAI"""
    
    # Initialize model - use a simple model path that should work
    # For now, let's skip the actual model loading and focus on the algorithm
    model = None
    tokenizer = None
    
    logger.info("Skipping model loading for now - focusing on algorithm design")
    
    # Test cases
    context = "The weather today is"
    continuation = " sunny and warm"
    full_text = context + continuation
    
    logger.info(f"Testing context: '{context}'")
    logger.info(f"Testing continuation: '{continuation}'")
    logger.info(f"Full text: '{full_text}'")
    
    # Approach 1: Echo mode with full text
    logger.info("\n=== Approach 1: Echo mode with full text ===")
    try:
        config1 = ov_genai.GenerationConfig()
        config1.echo = True
        config1.max_new_tokens = 0
        config1.logprobs = 5  # Get logprobs for tokens
        config1.do_sample = False
        
        result1 = model.generate(full_text, config1)
        logger.info(f"Result type: {type(result1)}")
        logger.info(f"Result attributes: {dir(result1)}")
        
        if hasattr(result1, 'scores'):
            logger.info(f"Scores: {result1.scores}")
        if hasattr(result1, 'texts'):
            logger.info(f"Texts: {result1.texts}")
            
    except Exception as e:
        logger.error(f"Approach 1 failed: {e}")
    
    # Approach 2: Generation mode from context
    logger.info("\n=== Approach 2: Generation mode from context ===")
    try:
        config2 = ov_genai.GenerationConfig()
        config2.echo = False
        config2.max_new_tokens = 5  # Generate a few tokens
        config2.logprobs = 5
        config2.do_sample = False
        config2.temperature = 0.0
        
        result2 = model.generate(context, config2)
        logger.info(f"Result type: {type(result2)}")
        logger.info(f"Result attributes: {dir(result2)}")
        
        if hasattr(result2, 'scores'):
            logger.info(f"Scores: {result2.scores}")
        if hasattr(result2, 'texts'):
            logger.info(f"Texts: {result2.texts}")
            
    except Exception as e:
        logger.error(f"Approach 2 failed: {e}")
    
    # Approach 3: Check tokenizer capabilities
    logger.info("\n=== Approach 3: Tokenizer analysis ===")
    try:
        encoded_context = tokenizer.encode(context)
        encoded_continuation = tokenizer.encode(continuation)
        encoded_full = tokenizer.encode(full_text)
        
        logger.info(f"Context tokens: {list(encoded_context)}")
        logger.info(f"Continuation tokens: {list(encoded_continuation)}")
        logger.info(f"Full text tokens: {list(encoded_full)}")
        
        # Check if continuation tokens appear at the end of full text tokens
        context_len = len(list(encoded_context))
        full_len = len(list(encoded_full))
        logger.info(f"Context length: {context_len}, Full length: {full_len}")
        
    except Exception as e:
        logger.error(f"Approach 3 failed: {e}")

if __name__ == "__main__":
    test_different_approaches()