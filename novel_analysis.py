import os
import re
import random
import json
import asyncio
import argparse
from pathlib import Path
from google import genai
from google.genai import types
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
# Configuration is loaded from .env file

BATCH_SIZE = 1  # Max concurrent API callsX

def extract_json(text):
    """
    Robustly extracts the last valid JSON object from a string that might
    contain multiple JSON blocks or extra chatter. Handles nested JSON correctly
    by filtering out objects that are contained within other objects.
    """
    results = [] # List of (obj, start_index, end_index)
    
    # Find all potential starting points of JSON objects
    for i in range(len(text)):
        if text[i] == '{':
            try:
                # Use raw_decode to find the first valid JSON object starting at this position
                obj, index = json.JSONDecoder().raw_decode(text[i:])
                # index is the number of characters consumed from text[i:]
                # so the absolute end index is i + index
                results.append((obj, i, i + index))
            except (json.JSONDecodeError, ValueError):
                continue
    
    if not results:
        return None
        
    # Filter out objects that are contained within others
    # We want top-level objects only.
    # An object A is contained in B if B.start <= A.start and B.end >= A.end (and not identical)
    final_candidates = []
    
    for i in range(len(results)):
        current_obj, start, end = results[i]
        is_contained = False
        
        for j in range(len(results)):
            if i == j: continue
            
            other_obj, other_start, other_end = results[j]
            
            if other_start <= start and other_end >= end:
                if not (other_start == start and other_end == end):
                     is_contained = True
                     break
        
        if not is_contained:
            final_candidates.append(current_obj)

    if not final_candidates:
        return None
    
    # Return the last one found from the top-level candidates
    return final_candidates[-1]

def preprocess_text(file_path):
    """
    Reads a text file and extracts a sample based on the rules.
    """
    encodings = ['utf-8', 'gb18030', 'gbk', 'utf-16', 'utf-16-le', 'utf-16-be']
    text = None
    
    for enc in encodings:
        try:
            # First try without ignoring errors to find the "best" match
            content = Path(file_path).read_text(encoding=enc)
            text = content
            break
        except (UnicodeDecodeError, Exception):
            continue
    
    if text is None:
        try:
            text = Path(file_path).read_text(encoding='utf-8', errors='ignore')
            print(f"Warning: Had to use utf-8 with ignore for {file_path}. Text might be garbled (Mojibake).")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    try:
        if not text:
            return ""

        return text

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

async def check_writing_quality(client, sample_text, model_version, write_quality_standard_content):
    """
    Lite check for writing quality based on the first 30k characters.
    """
    system_instruction = write_quality_standard_content
    user_message = f"Text Content:\n{sample_text}"
    max_retries = 4
    for attempt in range(max_retries):
        print(f"Attempt {attempt + 1} of {max_retries} for writing quality check...")
        try:
            response = await call_generate_content(client, model_version, user_message, system_instruction, enable_thinking=False)
            if not response.text:
                raise ValueError("Lite check: AI returned no text (possibly blocked by safety).")
                
            raw_text = response.text.strip()
            
            # Use the new robust extraction
            result = extract_json(raw_text)
            if result is None:
                raise ValueError(f"Could not extract valid JSON from response: {raw_text[:200]}...")
                
            return result
        except Exception as e:
            print(f"Lite check failed: {e}")
            error_str = str(e).lower()
            is_retryable = any(x in error_str for x in ["429", "rate limit", "500", "503", "504"])
            if attempt < max_retries - 1:
                if is_retryable:
                    wait_time = 15 * (attempt + 1) + random.uniform(0, 1)
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    wait_time = random.uniform(0, 1)
                    await asyncio.sleep(wait_time)
                    continue
            return {"writing_score": 0, "evidence_paragraph": f"Error: {e}"}
    return {"writing_score": 0, "evidence_paragraph": "Max retries reached"}

async def call_generate_content(client, model_version, user_message, system_instruction, enable_thinking=True):
    """
    Helper function to call the Gemini API with the specific configuration for full analysis.
    """
    config_params = {
        "system_instruction": system_instruction,
        "temperature": 0,
        "response_mime_type": "application/json",
        "safety_settings": [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
        ]
    }
    
    if enable_thinking:
        config_params["thinking_config"] = types.ThinkingConfig(thinking_level="high")

    return await client.aio.models.generate_content(
        model=model_version,
        contents=user_message,
        config=types.GenerateContentConfig(**config_params)
    )

async def analyze_file(file_path, semaphore, client, standard_content, write_quality_standard_content):
    """
    Sends the pre-processed text to the LLM for analysis using the new google-genai SDK.
    """
    async with semaphore:
        # Add jitter to stagger the burst of requests
        await asyncio.sleep(random.uniform(0.1, 2.0))
        
        text_content = preprocess_text(file_path)
        if not text_content:
            return None

        clean_name = re.sub(r'\.txt$', '', str(file_path).split('\\')[-1])
        model_version = os.environ.get("MODEL_VERSION", "gemini-3-flash-preview")

        # Lite check for writing quality to early exit
        if os.environ.get("ENABLE_QUICK_QUALITY_CHECK", "false").lower() == "true":
            try:
                sample_size = int(os.environ.get("SAMPLE_SIZE", 30000))
                sample_text = text_content[:sample_size]
                lite_check = await check_writing_quality(client, sample_text, model_version, write_quality_standard_content)
                
                rating = lite_check.get("rating", 0)
                dimension_rating = lite_check.get("dimension_rating", "")
                rating_paragraph = lite_check.get("rating_paragraph", "")

                if rating < 5.5:
                    return {
                        "file": clean_name,
                        "pass": False,
                        "rating": rating,
                        "dimension_rating": dimension_rating,
                        "rating_paragraph": rating_paragraph
                    }
            
            except Exception as e:
                # Error reporting with more raw context for debugging
                raw_context = raw_text if 'raw_text' in locals() else "N/A"
                # Limit context size but keep enough to see the mistake
                if len(raw_context) > 500:
                    raw_context = raw_context[:250] + "...[TRUNCATED]..." + raw_context[-250:]
                
                return {
                    "file": clean_name,
                    "error": f"JSON/API Error: {str(e)}",
                    "raw_output": raw_context
                }

        # 2. Proceed to full analysis if score >= 5.5
        system_instruction = standard_content
        
        user_message = f"Text Content:\n{text_content}"

        try:
            model_version = os.environ.get("MODEL_VERSION")
            
            # Simple retry logic for 429 Rate Limit and common transient errors
            max_retries = 4
            for attempt in range(max_retries):
                print(f"Attempt {attempt + 1} of {max_retries} for full analysis...")
                try:
                    response = await call_generate_content(client, model_version, user_message, system_instruction)
                    break # Success!
                except Exception as e:
                    error_str = str(e).lower()
                    is_retryable = any(x in error_str for x in ["429", "rate limit", "500", "503", "504"])
                    
                    if attempt < max_retries - 1:
                        if is_retryable:
                            # 15s per retry as requested
                            wait_time = 15 * (attempt + 1) + random.uniform(0, 1)
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            wait_time = random.uniform(0, 1)
                            await asyncio.sleep(wait_time)
                            continue
                    raise e # Re-raise if not retryable or max retries reached

            
            if not response.text:
                error_details = "AI returned no text."                
                raise ValueError(error_details)
            
            raw_text = response.text.strip()

            result = extract_json(raw_text)

            if result is None:
                # Try one more fallback: simple regex for when raw_decode might be too strict
                json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if json_match:
                    try:
                        # Clean trailing commas which raw_decode/loads hates
                        clean_text = re.sub(r',\s*([\]\}])', r'\1', json_match.group())
                        result = json.loads(clean_text, strict=False)
                    except:
                        pass
            
            if result is None:
                raise ValueError(f"Failed to extract valid JSON. Raw snippet: {raw_text[:200]}...")

            # Construct the final dictionary with 'file' at the beginning
            final_result = {"file": clean_name}
            final_result.update(result)
            final_result["usage"] = { 
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "candidates_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count
            }
            
            return final_result

        except Exception as e:
            # Error reporting with more raw context for debugging
            raw_context = raw_text if 'raw_text' in locals() else "N/A"
            # Limit context size but keep enough to see the mistake
            if len(raw_context) > 500:
                raw_context = raw_context[:250] + "...[TRUNCATED]..." + raw_context[-250:]
                
            return {
                "file": clean_name,
                "error": f"JSON/API Error: {str(e)}",
                "raw_output": raw_context
            }

async def analyze_file_pessimistic(file_path, semaphore, client, standard_content, write_quality_standard_content):
    """
    Calls analyze_file twice and returns the result with the lower rating.
    """
    # Run both analyses concurrently
    res1, res2 = await asyncio.gather(
        analyze_file(file_path, semaphore, client, standard_content, write_quality_standard_content),
        analyze_file(file_path, semaphore, client, standard_content, write_quality_standard_content)
    )
    
    # Selection logic
    if not res1: return res2
    if not res2: return res1
    
    if "error" in res1 and "error" not in res2: return res2
    if "error" in res2 and "error" not in res1: return res1
    
    try:
        r1 = float(res1.get("rating", 0))
        r2 = float(res2.get("rating", 0))
        return res1 if r1 <= r2 else res2
    except:
        return res1

async def main():
    print("--- Novel Analysis Script ---")
    
    # Read Standard from File
    standard_path_str, write_quality_standard_path_str = os.environ.get("STANDARD_PATH"), os.environ.get("WRITE_QUALITY_STANDARD_PATH")
    if not standard_path_str or not write_quality_standard_path_str:
        print("Error: STANDARD_PATH or WRITE_QUALITY_STANDARD_PATH environment variable not set.")
        return
        
    standard_path = Path(standard_path_str)
    write_quality_standard_path = Path(write_quality_standard_path_str)
    if not standard_path.exists() or not write_quality_standard_path.exists():
        print(f"Error: Standard file not found at {standard_path} or {write_quality_standard_path}")
        return
    
    standard_content = standard_path.read_text(encoding='utf-8', errors='ignore')
    write_quality_standard_content = write_quality_standard_path.read_text(encoding='utf-8', errors='ignore')
    print("Loaded analysis standard.")

    # Get API Key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = input("Enter your Google API Key: ").strip()
    
    # Initialize the new SDK client
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

    # Get Folder Path
    folder_path = os.environ.get("FOLDER_PATH")
    if not folder_path:
        folder_path = input("Please enter the folder path to analyze:").strip()

    if not os.path.isdir(folder_path):
        print(f"Invalid directory: {folder_path}")
        return

    # Find all txt files
    print(f"Scanning {folder_path} for .txt files...")
    all_files = list(Path(folder_path).rglob("*.txt"))
    print(f"Found {len(all_files)} files.")

    if not all_files:
        return

    # Semaphore for concurrency
    semaphore = asyncio.Semaphore(BATCH_SIZE)

    # Process files=
    if os.environ.get("ENABLE_DUAL_PASS", "false").lower() == "true":
        print(f"Starting analysis (Dual-Pass enabled: {len(all_files)} files)...")
        tasks = [analyze_file_pessimistic(f, semaphore, client, standard_content, write_quality_standard_content) for f in all_files]
    else:
        print(f"Starting analysis (Single-Pass enabled: {len(all_files)} files)...")
        tasks = [analyze_file(f, semaphore, client, standard_content, write_quality_standard_content) for f in all_files]
    results = []
    
    total_prompt_tokens = 0
    total_candidates_tokens = 0

    # --- INCREMENTAL SAVING SETUP ---
    output_dir_str = os.environ.get("OUTPUT_DIR")       
    output_dir = Path(output_dir_str)
    report_base = "analysis_report"
    report_ext = ".json"
    report_path = output_dir / f"{report_base}{report_ext}"
    counter = 1
    while report_path.exists():
        report_path = output_dir / f"{report_base}_{counter}{report_ext}"
        counter += 1
    
    print(f"Report will be saved to: {report_path.absolute()}")
    
    # Initialize empty JSON file
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2, ensure_ascii=False)

    
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Analyzing"):
        res = await f
        if res:
            # Directly append the result as it now contains the full JSON + file metadata
            results.append(res)
            
            # Handle token totals
            if "usage" in res:
                total_prompt_tokens += res["usage"]["prompt_tokens"]
                total_candidates_tokens += res["usage"]["candidates_tokens"]
            
            # INCREMENTAL SAVE
            try:
                # We overwrite the file with the current accumulated results
                # This ensures if the script stops, we have valid JSON up to this point
                with open(report_path, "w", encoding="utf-8") as f_out:
                    json.dump(results, f_out, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Warning: Failed to save progress: {e}")

    # Filter and Report
    passed_files = [r for r in results if r.get("pass") == True]
    
    print("\n\n" + "="*30)
    print(f"ANALYSIS COMPLETE. Found {len(passed_files)} matches out of {len(results)} files.")
    print(f"Total Usage: {total_prompt_tokens + total_candidates_tokens} tokens "
          f"(Input: {total_prompt_tokens}, Output: {total_candidates_tokens})")
    print("="*30 + "\n")

    # Sort Results by Rating (Descending)
    # We use .get("rating", 0) and handle potential non-numeric values
    def get_rating(item):
        val = item.get("rating", 0)
        try:
            return float(val) if val is not None else 0
        except:
            return 0
            
    results.sort(key=get_rating, reverse=True)

    # Final Save (Sorted)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    passed_files.sort(key=lambda x: x['rating'], reverse=True)
    for item in passed_files:
        raw_name = item['file'].split('\\')[-1]
        # Remove leading [...] brackets and spaces
        clean_name = re.sub(r'^\[.*?\]\s*', '', raw_name)
        # Remove .txt extension
        clean_name = re.sub(r'\.txt$', '', clean_name)
        print(f"File: {clean_name}")
        print("-" * 20)

    print(f"\nFull report saved to {report_path.absolute()}")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
