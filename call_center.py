#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

from datetime import datetime
from functools import reduce
from http import HTTPStatus
from io import BytesIO
from json import dumps, loads
from os import linesep, makedirs, path
from time import sleep
from typing import Dict, List, Tuple
import uuid
import helper
import rest_helper
import user_config_helper

# This should not change unless you switch to a new version of the Speech REST API.
SPEECH_TRANSCRIPTION_PATH = "/speechtotext/v3.2/transcriptions"

# These should not change unless you switch to a new version of the Cognitive Language REST API.
SENTIMENT_ANALYSIS_PATH = "/language/:analyze-text"
SENTIMENT_ANALYSIS_QUERY = "?api-version=2024-11-01"
CONVERSATION_ANALYSIS_PATH = "/language/analyze-conversations/jobs"

CONVERSATION_ANALYSIS_QUERY = "?api-version=2024-11-01"
CONVERSATION_SUMMARY_MODEL_VERSION = "latest"
CONVERSATION_PII_MODEL_VERSION = "2024-11-01-preview"

# How long to wait while polling batch transcription and conversation analysis status.
WAIT_SECONDS = 10


class TranscriptionPhrase(object):
    def __init__(self, id: int, text: str, itn: str, lexical: str, speaker_number: int, offset: str, offset_in_ticks: float):
        self.id = id
        self.text = text
        self.itn = itn
        self.lexical = lexical
        self.speaker_number = speaker_number
        self.offset = offset
        self.offset_in_ticks = offset_in_ticks


class SentimentAnalysisResult(object):
    def __init__(self, speaker_number: int, offset_in_ticks: float, document: Dict):
        self.speaker_number = speaker_number
        self.offset_in_ticks = offset_in_ticks
        self.document = document


class ConversationAnalysisSummaryItem(object):
    def __init__(self, aspect: str, summary: str):
        self.aspect = aspect
        self.summary = summary


class ConversationAnalysisPiiItem(object):
    def __init__(self, category: str, text: str):
        self.category = category
        self.text = text


class ConversationAnalysisForSimpleOutput(object):
    def __init__(self, summary: List[ConversationAnalysisSummaryItem], pii_analysis: List[List[ConversationAnalysisPiiItem]]):
        self.summary = summary
        self.pii_analysis = pii_analysis


# This needs to be serialized to JSON, so we use a Dict instead of a class.
def get_combined_redacted_content(channel: int) -> Dict:
    return {
        "channel": channel,
        "display": "",
        "itn": "",
        "lexical": ""
    }


def get_speaker_segregated_phrases(transcription: Dict) -> List[Dict]:
    """
    Create speaker-segregated conversation text organized by format type.
    Groups consecutive utterances by speaker and formats them as a conversation.
    
    Returns:
        List of dictionaries with channel and text in multiple formats
    """
    recognized_phrases = transcription.get("recognizedPhrases", [])
    
    if not recognized_phrases:
        return [{
            "channel": 0,
            "lexical": "",
            "itn": "",
            "maskedITN": "",
            "display": ""
        }]
    
    # Build conversation lines with speaker labels
    conversation_lines_lexical = []
    conversation_lines_itn = []
    conversation_lines_masked_itn = []
    conversation_lines_display = []
    
    current_speaker = None
    current_text_lexical = []
    current_text_itn = []
    current_text_masked_itn = []
    current_text_display = []
    
    for phrase in recognized_phrases:
        speaker = phrase.get("speaker")
        
        # Get the best transcription from nBest
        if phrase.get("nBest") and len(phrase["nBest"]) > 0:
            best = phrase["nBest"][0]
            
            # If same speaker, accumulate text
            if speaker == current_speaker:
                current_text_lexical.append(best.get("lexical", ""))
                current_text_itn.append(best.get("itn", ""))
                current_text_masked_itn.append(best.get("maskedITN", ""))
                current_text_display.append(best.get("display", ""))
            else:
                # Different speaker - save previous speaker's text if exists
                if current_speaker is not None:
                    conversation_lines_lexical.append(f"Speaker {current_speaker}: {' '.join(current_text_lexical)}")
                    conversation_lines_itn.append(f"Speaker {current_speaker}: {' '.join(current_text_itn)}")
                    conversation_lines_masked_itn.append(f"Speaker {current_speaker}: {' '.join(current_text_masked_itn)}")
                    conversation_lines_display.append(f"Speaker {current_speaker}: {' '.join(current_text_display)}")
                
                # Start new speaker
                current_speaker = speaker
                current_text_lexical = [best.get("lexical", "")]
                current_text_itn = [best.get("itn", "")]
                current_text_masked_itn = [best.get("maskedITN", "")]
                current_text_display = [best.get("display", "")]
    
    # Add the last speaker's text
    if current_speaker is not None:
        conversation_lines_lexical.append(f"Speaker {current_speaker}: {' '.join(current_text_lexical)}")
        conversation_lines_itn.append(f"Speaker {current_speaker}: {' '.join(current_text_itn)}")
        conversation_lines_masked_itn.append(f"Speaker {current_speaker}: {' '.join(current_text_masked_itn)}")
        conversation_lines_display.append(f"Speaker {current_speaker}: {' '.join(current_text_display)}")
    
    # Get channel from first phrase (assuming single channel for now)
    channel = recognized_phrases[0].get("channel", 0) if recognized_phrases else 0
    
    return [{
        "channel": channel,
        "lexical": "\n".join(conversation_lines_lexical),
        "itn": "\n".join(conversation_lines_itn),
        "maskedITN": "\n".join(conversation_lines_masked_itn),
        "display": "\n".join(conversation_lines_display)
    }]


def create_transcription(user_config: helper.Read_Only_Dict) -> str:
    uri = f"https://{user_config['speech_endpoint']}{SPEECH_TRANSCRIPTION_PATH}"

    # Create Transcription API JSON request sample and schema:
    # https://westus.dev.cognitive.microsoft.com/docs/services/speech-to-text-api-v3-0/operations/CreateTranscription
    # Notes:
    # - locale and displayName are required (or languageIdentification for multi-language).
    # - diarizationEnabled should only be used with mono audio input.
    
    properties = {
        "diarizationEnabled": not user_config["use_stereo_audio"],
        "wordLevelTimestampsEnabled": True,
        "punctuationMode": "DictatedAndAutomatic",
        "profanityFilterMode": "Masked",
        "timeToLive": "PT30M"
    }
    
    content = {
        "contentUrls": [user_config["input_audio_url"]],
        "properties": properties,
        "displayName": f"call_center_{datetime.now()}",
    }
    
    # Use language identification if candidate locales are provided, otherwise use single locale
    if "candidate_locales" in user_config and user_config["candidate_locales"]:
        properties["languageIdentification"] = {
            "candidateLocales": user_config["candidate_locales"],
            "languageIdentificationMode": user_config.get("language_identification_mode", "Continuous")
        }
        # When using language identification, set locale to the first candidate as a fallback
        content["locale"] = user_config["candidate_locales"][0]
    else:
        content["locale"] = user_config["locale"]
    
    # Add Whisper model if specified
    if "whisper_model" in user_config and user_config["whisper_model"]:
        content["model"] = user_config["whisper_model"]
        # For Whisper models, use displayFormWordLevelTimestampsEnabled instead of wordLevelTimestampsEnabled
        properties["displayFormWordLevelTimestampsEnabled"] = True
        del properties["wordLevelTimestampsEnabled"]
        # Whisper doesn't support punctuationMode
        del properties["punctuationMode"]

    response = rest_helper.send_post(uri=uri, content=content, key=user_config["speech_subscription_key"],
                                     expected_status_codes=[HTTPStatus.CREATED])

    # Create Transcription API JSON response sample and schema:
    # https://westus.dev.cognitive.microsoft.com/docs/services/speech-to-text-api-v3-0/operations/CreateTranscription
    transcription_uri = response["json"]["self"]
    # The transcription ID is at the end of the transcription URI.
    transcription_id = transcription_uri.split("/")[-1]
    # Verify the transcription ID is a valid GUID.
    try:
        uuid.UUID(transcription_id)
        return transcription_id
    except ValueError:
        raise Exception(f"Unable to parse response from Create Transcription API:{linesep}{response['text']}")


def get_transcription_status(transcription_id: str, user_config: helper.Read_Only_Dict) -> bool:
    uri = f"https://{user_config['speech_endpoint']}{SPEECH_TRANSCRIPTION_PATH}/{transcription_id}"
    response = rest_helper.send_get(uri=uri, key=user_config["speech_subscription_key"], expected_status_codes=[HTTPStatus.OK])
    if "failed" == response["json"]["status"].lower():
        raise Exception(f"Unable to transcribe audio input. Response:{linesep}{response['text']}")
    else:
        return "succeeded" == response["json"]["status"].lower()


def wait_for_transcription(transcription_id: str, user_config: helper.Read_Only_Dict) -> None:
    done = False
    while not done:
        print(f"Waiting {WAIT_SECONDS} seconds for transcription to complete.")
        sleep(WAIT_SECONDS)
        done = get_transcription_status(transcription_id, user_config=user_config)


def get_transcription_files(transcription_id: str, user_config: helper.Read_Only_Dict) -> Dict:
    uri = f"https://{user_config['speech_endpoint']}{SPEECH_TRANSCRIPTION_PATH}/{transcription_id}/files"
    response = rest_helper.send_get(uri=uri, key=user_config["speech_subscription_key"], expected_status_codes=[HTTPStatus.OK])
    return response["json"]


def get_transcription_uri(transcription_files: Dict, user_config: helper.Read_Only_Dict) -> str:
    # Get Transcription Files JSON response sample and schema:
    # https://westus.dev.cognitive.microsoft.com/docs/services/speech-to-text-api-v3-0/operations/GetTranscriptionFiles
    value = next(filter(lambda value: "transcription" == value["kind"].lower(), transcription_files["values"]), None)
    if value is None:
        raise Exception(f"Unable to parse response from Get Transcription Files API:{linesep}{transcription_files['text']}")
    return value["links"]["contentUrl"]


def get_transcription(transcription_uri: str) -> Dict:
    response = rest_helper.send_get(uri=transcription_uri, key="", expected_status_codes=[HTTPStatus.OK])
    return response["json"]


def get_transcription_phrases(transcription: Dict, user_config: helper.Read_Only_Dict) -> List[TranscriptionPhrase]:
    def helper(id_and_phrase: Tuple[int, Dict]) -> TranscriptionPhrase:
        (id, phrase) = id_and_phrase
        best = phrase["nBest"][0]
        speaker_number: int
        # If the user specified stereo audio, and therefore we turned off diarization,
        # only the channel property is present.
        # Note: Channels are numbered from 0. Speakers are numbered from 1.
        if "speaker" in phrase:
            speaker_number = phrase["speaker"] - 1
        elif "channel" in phrase:
            speaker_number = phrase["channel"]
        else:
            raise Exception(f"nBest item contains neither channel nor speaker attribute.{linesep}{best}")
        return TranscriptionPhrase(id, best["display"], best["itn"], best["lexical"], speaker_number,
                                   phrase["offset"], phrase["offsetInTicks"])
    # For stereo audio, the phrases are sorted by channel number, so resort them by offset.
    return list(map(helper, enumerate(transcription["recognizedPhrases"])))


def delete_transcription(transcription_id: str, user_config: helper.Read_Only_Dict) -> None:
    uri = f"https://{user_config['speech_endpoint']}{SPEECH_TRANSCRIPTION_PATH}/{transcription_id}"
    rest_helper.send_delete(uri=uri, key=user_config["speech_subscription_key"], expected_status_codes=[HTTPStatus.NO_CONTENT])


def get_sentiments_helper(documents: List[Dict], user_config: helper.Read_Only_Dict) -> Dict:
    uri = f"https://{user_config['language_endpoint']}{SENTIMENT_ANALYSIS_PATH}{SENTIMENT_ANALYSIS_QUERY}"
    content = {
        "kind": "SentimentAnalysis",
        "analysisInput": {"documents": documents},
    }
    response = rest_helper.send_post(uri=uri, content=content, key=user_config["language_subscription_key"],
                                     expected_status_codes=[HTTPStatus.OK])
    return response["json"]["results"]["documents"]


def get_sentiment_analysis(phrases: List[TranscriptionPhrase], user_config: helper.Read_Only_Dict) -> List[SentimentAnalysisResult]:
    retval: List[SentimentAnalysisResult] = []
    # Create a map of phrase ID to phrase data so we can retrieve it later.
    phrase_data: Dict = {}
    # Convert each transcription phrase to a "document" as expected by the sentiment analysis REST API.
    # Include a counter to use as a document ID.
    documents: List[Dict] = []
    for phrase in phrases:
        phrase_data[phrase.id] = (phrase.speaker_number, phrase.offset_in_ticks)
        documents.append({
            "id": phrase.id,
            "language": user_config["language"],
            "text": phrase.text,
        })
    # We can only analyze sentiment for 10 documents per request.
    # Get the sentiments for each chunk of documents.
    result_chunks = list(map(lambda xs: get_sentiments_helper(xs, user_config), helper.chunk(documents, 10)))
    for result_chunk in result_chunks:
        for document in result_chunk:
            retval.append(SentimentAnalysisResult(
                phrase_data[int(document["id"])][0],
                phrase_data[int(document["id"])][1],
                document
            ))
    return retval


def get_sentiments_for_simple_output(sentiment_analysis_results: List[SentimentAnalysisResult]) -> List[str]:
    sorted_by_offset = sorted(sentiment_analysis_results, key=lambda x: x.offset_in_ticks)
    return list(map(lambda result: result.document["sentiment"], sorted_by_offset))


def get_sentiment_confidence_scores(sentiment_analysis_results: List[SentimentAnalysisResult]) -> List[Dict]:
    sorted_by_offset = sorted(sentiment_analysis_results, key=lambda x: x.offset_in_ticks)
    return list(map(lambda result: result.document["confidenceScores"], sorted_by_offset))


def merge_sentiment_confidence_scores_into_transcription(transcription: Dict, sentiment_confidence_scores: List[Dict]) -> Dict:
    for id, phrase in enumerate(transcription["recognizedPhrases"]):
        for best_item in phrase["nBest"]:
            best_item["sentiment"] = sentiment_confidence_scores[id]
    return transcription


def transcription_phrases_to_conversation_items(phrases: List[TranscriptionPhrase]) -> List[Dict]:
    return [{
        "id": phrase.id,
        "text": phrase.text,
        "itn": phrase.itn,
        "lexical": phrase.lexical,
        # The first person to speak is probably the agent.
        "role": "Agent" if 0 == phrase.speaker_number else "Customer",
        "participantId": phrase.speaker_number
    } for phrase in phrases]


def request_conversation_analysis(conversation_items: List[Dict], user_config: helper.Read_Only_Dict) -> str:
    uri = f"https://{user_config['language_endpoint']}{CONVERSATION_ANALYSIS_PATH}{CONVERSATION_ANALYSIS_QUERY}"
    content = {
        "displayName": f"call_center_{datetime.now()}",
        "analysisInput": {
            "conversations": [{
                "id": "conversation1",
                "language": user_config["language"],
                "modality": "transcript",
                "conversationItems": conversation_items,
            }],
        },
        "tasks": [
            {
                "taskName": "summary_1",
                "kind": "ConversationalSummarizationTask",
                "parameters": {
                    "modelVersion": CONVERSATION_SUMMARY_MODEL_VERSION,
                    "summaryAspects": [
                        "Issue",
                        "Resolution"
                    ],
                }
            },
            {
                "taskName": "PII_1",
                "kind": "ConversationalPIITask",
                "parameters": {
                    "piiCategories": [
                        "All",
                    ],
                    "includeAudioRedaction": False,
                    "redactionSource": "text",
                    "modelVersion": CONVERSATION_PII_MODEL_VERSION,
                    "loggingOptOut": False
                }
            }
        ]
    }
    response = rest_helper.send_post(uri=uri, content=content, key=user_config["language_subscription_key"],
                                     expected_status_codes=[HTTPStatus.ACCEPTED])
    return response["headers"]["operation-location"]


def get_conversation_analysis_status(conversation_analysis_url: str, user_config: helper.Read_Only_Dict) -> bool:
    response = rest_helper.send_get(uri=conversation_analysis_url, key=user_config["language_subscription_key"],
                                    expected_status_codes=[HTTPStatus.OK])
    if "failed" == response["json"]["status"].lower():
        raise Exception(f"Unable to analyze conversation. Response:{linesep}{response['text']}")
    else:
        return "succeeded" == response["json"]["status"].lower()


def wait_for_conversation_analysis(conversation_analysis_url: str, user_config: helper.Read_Only_Dict) -> None:
    done = False
    while not done:
        print(f"Waiting {WAIT_SECONDS} seconds for conversation analysis to complete.")
        sleep(WAIT_SECONDS)
        done = get_conversation_analysis_status(conversation_analysis_url, user_config=user_config)


def get_conversation_analysis(conversation_analysis_url: str, user_config: helper.Read_Only_Dict) -> Dict:
    response = rest_helper.send_get(uri=conversation_analysis_url, key=user_config["language_subscription_key"],
                                    expected_status_codes=[HTTPStatus.OK])
    return response["json"]


def get_conversation_analysis_for_simple_output(conversation_analysis: Dict,
                                                user_config: helper.Read_Only_Dict) -> ConversationAnalysisForSimpleOutput:
    tasks = conversation_analysis["tasks"]["items"]

    summary_task = next(filter(lambda task: "summary_1" == task["taskName"], tasks), None)
    if summary_task is None:
        raise Exception(f"Unable to parse response from Get Conversation Analysis API. Summary task missing. Response:"
                        f"{linesep}{conversation_analysis}")
    conversation = summary_task["results"]["conversations"][0]
    summary_items = list(map(lambda summary: ConversationAnalysisSummaryItem(summary["aspect"], summary["text"]),
                             conversation["summaries"]))

    pii_task = next(filter(lambda task: "PII_1" == task["taskName"], tasks), None)
    if pii_task is None:
        raise Exception(f"Unable to parse response from Get Conversation Analysis API. PII task missing. Response:"
                        f"{linesep}{conversation_analysis}")
    conversation = pii_task["results"]["conversations"][0]

    # Avoid complex list comprehension that causes flake8 errors
    pii_items = []
    for conversation_item in conversation["conversationItems"]:
        item_entities = []
        for entity in conversation_item["entities"]:
            item_entities.append(ConversationAnalysisPiiItem(entity["category"], entity["text"]))
        pii_items.append(item_entities)

    return ConversationAnalysisForSimpleOutput(summary_items, pii_items)


def get_simple_output(phrases: List[TranscriptionPhrase], sentiments: List[str],
                      conversation_analysis: ConversationAnalysisForSimpleOutput) -> str:
    result = ""
    for index, phrase in enumerate(phrases):
        result += f"Phrase: {phrase.text}{linesep}"
        result += f"Speaker: {phrase.speaker_number}{linesep}"
        if index < len(sentiments):
            result += f"Sentiment: {sentiments[index]}{linesep}"
        if index < len(conversation_analysis.pii_analysis):
            if len(conversation_analysis.pii_analysis[index]) > 0:
                entities = reduce(
                    lambda acc, entity: f"{acc}    Category: {entity.category}. Text: {entity.text}.{linesep}",
                    conversation_analysis.pii_analysis[index], ""
                )
                result += f"Recognized entities (PII):{linesep}{entities}"
            else:
                result += f"Recognized entities (PII): none.{linesep}"
        result += linesep
        result += reduce(
            lambda acc, item: f"{acc}    {item.aspect}: {item.summary}.{linesep}",
            conversation_analysis.summary, f"Conversation summary:{linesep}"
        )
    return result


def print_simple_output(phrases: List[TranscriptionPhrase], sentiment_analysis_results: List[SentimentAnalysisResult],
                        conversation_analysis: Dict, user_config: helper.Read_Only_Dict) -> None:
    sentiments = get_sentiments_for_simple_output(sentiment_analysis_results)
    conversation = get_conversation_analysis_for_simple_output(conversation_analysis, user_config)
    print(get_simple_output(phrases, sentiments, conversation))


def get_conversation_analysis_for_full_output(phrases: List[TranscriptionPhrase], conversation_analysis: Dict) -> Dict:
    # Get the conversation summary and conversation PII analysis task results.
    tasks = conversation_analysis["tasks"]["items"]
    conversation_summary_results = next(filter(lambda task: "summary_1" == task["taskName"], tasks))["results"]
    conversation_pii_results = next(filter(lambda task: "PII_1" == task["taskName"], tasks))["results"]
    # There should be only one conversation.
    conversation = conversation_pii_results["conversations"][0]
    # Order conversation items by ID so they match the order of the transcription phrases.
    conversation["conversationItems"] = sorted(conversation["conversationItems"], key=lambda item: int(item["id"]))
    combined_redacted_content = [get_combined_redacted_content(0), get_combined_redacted_content(1)]
    for index, conversation_item in enumerate(conversation["conversationItems"]):
        # Get the channel and offset for this conversation item from the corresponding transcription phrase.
        channel = phrases[index].speaker_number
        # Add channel and offset to conversation item JsonElement.
        conversation_item["channel"] = channel
        conversation_item["offset"] = phrases[index].offset
        # Get the text, lexical, and itn fields from redacted content, and append them to the combined redacted content
        redacted_content = conversation_item["redactedContent"]
        combined_redacted_content[channel]["display"] += f"{redacted_content['text']} "
        combined_redacted_content[channel]["lexical"] += f"{redacted_content['lexical']} "
        combined_redacted_content[channel]["itn"] += f"{redacted_content['itn']} "
    return {
        "conversationSummaryResults": conversation_summary_results,
        "conversationPiiResults": {
            "combinedRedactedContent": combined_redacted_content,
            "conversations": [conversation]
        }
    }


def print_full_output(output_file_path: str, transcription: Dict, sentiment_confidence_scores: List[Dict],
                      phrases: List[TranscriptionPhrase], conversation_analysis: Dict) -> None:
    results = {
        "transcription": merge_sentiment_confidence_scores_into_transcription(transcription, sentiment_confidence_scores),
        "conversationAnalyticsResults": get_conversation_analysis_for_full_output(phrases, conversation_analysis)
    }
    # Create output directory if it doesn't exist
    output_dir = path.dirname(output_file_path)
    if output_dir and not path.exists(output_dir):
        makedirs(output_dir)
    with open(output_file_path, mode="w", newline="") as f:
        f.write(dumps(results, indent=2))


def upload_to_blob_storage(container_url: str, blob_name: str, content: str) -> bool:
    """Upload JSON content to Azure Blob Storage."""
    try:
        from azure.storage.blob import BlobClient
    except ImportError:
        raise ImportError(
            "Azure Storage Blob library is required for blob upload.\n"
            "Install it with: pip install azure-storage-blob"
        )
    
    try:
        # Extract SAS token from container URL if present
        if '?' in container_url:
            base_url = container_url.split('?')[0]
            sas_token = '?' + container_url.split('?')[1]
        else:
            base_url = container_url
            sas_token = ""
        
        # Construct blob URL
        blob_url = f"{base_url.rstrip('/')}/{blob_name}{sas_token}"
        
        # Create blob client and upload
        blob_client = BlobClient.from_blob_url(blob_url)
        content_bytes = content.encode('utf-8')
        blob_client.upload_blob(BytesIO(content_bytes), overwrite=True)
        
        return True
    except Exception as e:
        print(f"Failed to upload to blob storage: {str(e)}")
        return False


def export_speaker_conversation_to_txt(transcription: Dict, json_file_path: str) -> None:
    """Export speaker-segregated conversation to a single plain text file with all formats."""
    if not transcription.get("combinedRecognizedPhrases"):
        return
    
    # Get base filename without extension
    base_path = json_file_path.rsplit('.', 1)[0]
    txt_file_path = f"{base_path}.txt"
    
    # Get the speaker-segregated phrases
    combined_phrases = transcription["combinedRecognizedPhrases"][0]
    
    formats = [
        ('LEXICAL (Raw Text)', 'lexical'),
        ('ITN (Inverse Text Normalization)', 'itn'),
        ('MASKED ITN (Profanity Masked)', 'maskedITN'),
        ('DISPLAY (Final Output)', 'display')
    ]
    
    with open(txt_file_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SPEAKER-SEGREGATED CONVERSATION\n")
        f.write("=" * 80 + "\n\n")
        
        for format_title, format_key in formats:
            if format_key in combined_phrases:
                f.write("-" * 80 + "\n")
                f.write(f"{format_title}\n")
                f.write("-" * 80 + "\n")
                f.write(combined_phrases[format_key])
                f.write("\n\n")
    
    print(f"  ✓ Exported speaker conversation to: {path.basename(txt_file_path)}")


def list_blob_files(container_url: str) -> List[str]:
    """List all audio files in the blob container."""
    try:
        from azure.storage.blob import ContainerClient
    except ImportError:
        raise ImportError(
            "Azure Storage Blob library is required for container batch processing.\n"
            "Install it with: pip install azure-storage-blob"
        )
    
    audio_extensions = ('.wav', '.mp3', '.ogg', '.flac', '.m4a', '.wma', '.aac')
    audio_files = []
    
    try:
        container_client = ContainerClient.from_container_url(container_url)
        blob_list = container_client.list_blobs()
        
        # Extract SAS token from container URL if present
        sas_token = ""
        if '?' in container_url:
            base_url = container_url.split('?')[0]
            sas_token = '?' + container_url.split('?')[1]
        else:
            base_url = container_url
        
        for blob in blob_list:
            if blob.name.lower().endswith(audio_extensions):
                # Construct blob URL with SAS token
                blob_url = f"{base_url.rstrip('/')}/{blob.name}{sas_token}"
                audio_files.append(blob_url)
        
        return audio_files
    except Exception as e:
        raise Exception(f"Failed to list blobs in container: {str(e)}")


def process_single_audio(audio_url: str, user_config: helper.Read_Only_Dict, 
                        output_file_path: str = None) -> Dict:
    """Process a single audio file and return transcription."""
    # Create a temporary config for this audio file
    temp_config = dict(user_config)
    temp_config["input_audio_url"] = audio_url
    temp_config = helper.Read_Only_Dict(temp_config)
    
    transcription_id = create_transcription(temp_config)
    wait_for_transcription(transcription_id, temp_config)
    print(f"Transcription ID: {transcription_id}")
    transcription_files = get_transcription_files(transcription_id, temp_config)
    transcription_uri = get_transcription_uri(transcription_files, temp_config)
    transcription = get_transcription(transcription_uri)
    
    # Sort phrases by offset
    transcription["recognizedPhrases"] = sorted(
        transcription["recognizedPhrases"],
        key=lambda phrase: phrase["offsetInTicks"]
    )
    
    # Replace original combinedRecognizedPhrases with speaker-segregated conversation
    transcription["combinedRecognizedPhrases"] = get_speaker_segregated_phrases(transcription)
    
    phrases = get_transcription_phrases(transcription, temp_config)
    
    # Perform conversation analysis if language credentials provided
    if temp_config.get("language_subscription_key") and temp_config.get("language_endpoint"):
        sentiment_analysis_results = get_sentiment_analysis(phrases, temp_config)
        sentiment_confidence_scores = get_sentiment_confidence_scores(sentiment_analysis_results)
        conversation_items = transcription_phrases_to_conversation_items(phrases)
        conversation_analysis_url = request_conversation_analysis(conversation_items, temp_config)
        wait_for_conversation_analysis(conversation_analysis_url, temp_config)
        conversation_analysis = get_conversation_analysis(conversation_analysis_url, temp_config)
        
        if output_file_path:
            print_full_output(output_file_path, transcription, sentiment_confidence_scores, 
                            phrases, conversation_analysis)
            
            # Upload to blob storage if output container URL is provided
            if temp_config.get("output_container_url"):
                blob_name = path.basename(output_file_path)
                results = {
                    "transcription": merge_sentiment_confidence_scores_into_transcription(transcription, sentiment_confidence_scores),
                    "conversationAnalyticsResults": get_conversation_analysis_for_full_output(phrases, conversation_analysis)
                }
                json_content = dumps(results, indent=2)
                if upload_to_blob_storage(temp_config["output_container_url"], blob_name, json_content):
                    print(f"  ✓ Uploaded to blob storage: {blob_name}")
                else:
                    print(f"  ⚠️  Failed to upload to blob storage: {blob_name}")
    else:
        # Transcription only
        if output_file_path:
            output_dir = path.dirname(output_file_path)
            if output_dir and not path.exists(output_dir):
                makedirs(output_dir)
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(dumps(transcription, indent=2))
            
            # Export speaker-segregated conversation to plain text file
            if "combinedRecognizedPhrases" in transcription and transcription["combinedRecognizedPhrases"]:
                export_speaker_conversation_to_txt(transcription, output_file_path)
        
        # Upload to blob storage if output container URL is provided
        if temp_config.get("output_container_url") and output_file_path:
            blob_name = path.basename(output_file_path)
            json_content = dumps(transcription, indent=2)
            if upload_to_blob_storage(temp_config["output_container_url"], blob_name, json_content):
                print(f"  ✓ Uploaded to blob storage: {blob_name}")
            else:
                print(f"  ⚠️  Failed to upload to blob storage: {blob_name}")
    
    # Clean up transcription
    delete_transcription(transcription_id, temp_config)
    
    return transcription


def process_container_batch(user_config: helper.Read_Only_Dict) -> None:
    """Process all audio files in a blob container."""
    container_url = user_config["container_url"]
    output_folder = user_config["output_folder"]
    
    print(f"Listing audio files in container: {container_url}")
    audio_files = list_blob_files(container_url)
    
    if not audio_files:
        print("No audio files found in container.")
        return
    
    print(f"Found {len(audio_files)} audio file(s) to process.\n")
    
    # Create output folder if it doesn't exist
    if not path.exists(output_folder):
        makedirs(output_folder)
    
    success_count = 0
    failed_count = 0
    
    for index, audio_url in enumerate(audio_files, 1):
        # Extract filename from URL (remove SAS query parameters for display)
        url_path = audio_url.split('?')[0] if '?' in audio_url else audio_url
        filename = url_path.split('/')[-1]
        base_name = path.splitext(filename)[0]
        output_file = path.join(output_folder, f"{base_name}_transcription.json")
        
        print(f"[{index}/{len(audio_files)}] Processing: {filename}")
        print(f"  Audio URL: {audio_url}")
        
        try:
            process_single_audio(audio_url, user_config, output_file)
            print(f"✓ Successfully processed: {filename}")
            print(f"  Output saved to: {output_file}\n")
            success_count += 1
        except Exception as e:
            print(f"✗ Failed to process {filename}: {str(e)}\n")
            failed_count += 1
    
    print(f"\n=== Batch Processing Complete ===")
    print(f"Total files: {len(audio_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")


def run() -> None:
    usage = """python call_center.py [...]

  HELP
    --help                          Show this help and stop.

  ENVIRONMENT VARIABLES (.env file support)
    You can store credentials in a .env file instead of passing them via command line.
    Create a .env file in the same directory with the following format:
    
      SPEECH_KEY=your_speech_key
      SPEECH_REGION=your_region
      LANGUAGE_KEY=your_language_key (optional)
      LANGUAGE_ENDPOINT=your_language_endpoint (optional)
    
    Command line arguments override .env values.
    See .env.example for a template.

  CONNECTION
    --speechKey KEY                 Your Azure Speech service subscription key.
                                    Can be set via SPEECH_KEY in .env file.
                                    Command line value takes priority over .env.
                                    Required unless --jsonInput is present.
    --speechRegion REGION           Your Azure Speech service region.
                                    Can be set via SPEECH_REGION in .env file.
                                    Command line value takes priority over .env.
                                    Required unless --jsonInput is present.
                                    Examples: westus, eastus, centralindia
    --useTextAnalytics              Enable conversation analytics (sentiment, summary, PII).
                                    If not specified, only transcription is performed.
                                    Requires --languageKey and --languageEndpoint (or .env equivalents).
    --languageKey KEY               Your Azure Cognitive Language subscription key.
                                    Can be set via LANGUAGE_KEY in .env file.
                                    Command line value takes priority over .env.
                                    Only used if --useTextAnalytics is specified.
    --languageEndpoint ENDPOINT     Your Azure Cognitive Language endpoint.
                                    Can be set via LANGUAGE_ENDPOINT in .env file.
                                    Command line value takes priority over .env.
                                    Only used if --useTextAnalytics is specified.

  LANGUAGE
    --language LANGUAGE             The language to use for sentiment analysis and conversation analysis.
                                    This should be a two-letter ISO 639-1 code.
                                    Default: en
    --locale LOCALE                 The locale to use for batch transcription of audio (single language mode).
                                    Default: en-US
                                    Ignored if --candidateLocales is provided.
    --candidateLocales LOCALES      Comma-separated list of locales for multi-language detection.
                                    Can be set via CANDIDATE_LOCALES in .env file.
                                    Command line value takes priority over .env.
                                    Example: en-US,es-ES,fr-FR
                                    If provided, enables automatic language identification.
    --languageIdMode MODE           Language identification mode: 'Single' or 'Continuous'.
                                    Single: Detect one language for entire audio.
                                    Continuous: Detect language switches during call.
                                    Default: Continuous

  MODEL
    --whisperModel MODEL_URI        Use OpenAI Whisper model for transcription.
                                    Provide the full model URI from your Speech region.
                                    Example: https://centralindia.api.cognitive.microsoft.com/speechtotext/models/base/<model-id>
                                    Note: Whisper returns display-only results (no lexical/ITN forms).
                                    Use 'GET /speechtotext/models/base' API to list available Whisper models.

  INPUT
    --input URL                     Input audio from URL. Required unless --jsonInput or --containerUrl is present.
    --jsonInput FILE                Input JSON Speech batch transcription result from FILE. Overrides --input.
    --containerUrl URL              Azure Blob Storage container URL with SAS token for batch processing.
                                    All audio files in the container will be processed.
                                    Can be set via CONTAINER_URL in .env file.
                                    Command line value takes priority over .env.
                                    Example: https://account.blob.core.windows.net/container?sas_token
                                    Requires: pip install azure-storage-blob
    --stereo                        Use stereo audio format.
                                    If this is not present, mono is assumed.

  OUTPUT
    --output FILE                   Output phrase list and conversation summary to text file (single file mode).
    --outputFolder FOLDER           Output folder for batch processing results (used with --containerUrl).
                                    Can be set via OUTPUT_FOLDER in .env file.
                                    Command line value takes priority over .env.
                                    Default: output
    --outputContainerUrl URL        Azure Blob Storage container URL with SAS token for uploading results.
                                    Results will be uploaded to this container in addition to local storage.
                                    Can be set via OUTPUT_CONTAINER_URL in .env file.
                                    Command line value takes priority over .env.
                                    Example: https://account.blob.core.windows.net/results?sas_token
                                    Requires write permissions on the SAS token.
                                    Requires: pip install azure-storage-blob
"""

    if user_config_helper.cmd_option_exists("--help"):
        print(usage)
    else:
        user_config = user_config_helper.user_config_from_args(usage)
        
        # Check if batch processing mode (container URL provided)
        if user_config.get("container_url"):
            process_container_batch(user_config)
            return
        
        transcription: Dict
        transcription_id: str
        if user_config["input_file_path"] is not None:
            with open(user_config["input_file_path"], mode="r") as f:
                transcription = loads(f.read())
        elif user_config["input_audio_url"] is not None:
            # How to use batch transcription:
            # https://github.com/MicrosoftDocs/azure-docs/blob/main/articles/cognitive-services/Speech-Service/batch-transcription.md
            transcription_id = create_transcription(user_config)
            wait_for_transcription(transcription_id, user_config)
            print(f"Transcription ID: {transcription_id}")
            transcription_files = get_transcription_files(transcription_id, user_config)
            transcription_uri = get_transcription_uri(transcription_files, user_config)
            print(f"Transcription URI: {transcription_uri}")
            transcription = get_transcription(transcription_uri)
        else:
            raise Exception(f"Missing input audio URL.{linesep}{usage}")
        # For stereo audio, the phrases are sorted by channel number, so resort them by offset.
        transcription["recognizedPhrases"] = sorted(
            transcription["recognizedPhrases"],
            key=lambda phrase: phrase["offsetInTicks"]
        )
        phrases = get_transcription_phrases(transcription, user_config)
        
        # Only perform conversation analysis if language key and endpoint are provided
        if user_config.get("language_subscription_key") and user_config.get("language_endpoint"):
            sentiment_analysis_results = get_sentiment_analysis(phrases, user_config)
            sentiment_confidence_scores = get_sentiment_confidence_scores(sentiment_analysis_results)
            conversation_items = transcription_phrases_to_conversation_items(phrases)
            # NOTE: Conversation summary is currently in gated public preview. You can sign up here:
            # https://aka.ms/applyforconversationsummarization/
            conversation_analysis_url = request_conversation_analysis(conversation_items, user_config)
            wait_for_conversation_analysis(conversation_analysis_url, user_config)
            conversation_analysis = get_conversation_analysis(conversation_analysis_url, user_config)
            print_simple_output(phrases, sentiment_analysis_results, conversation_analysis, user_config)
            if user_config["output_file_path"] is not None:
                print_full_output(
                    user_config["output_file_path"],
                    transcription,
                    sentiment_confidence_scores,
                    phrases,
                    conversation_analysis
                )
        else:
            # Transcription only - no conversation analysis
            print("\nTranscription Results (no conversation analysis):")
            for phrase in phrases:
                print(f"\nSpeaker {phrase.speaker_number}: {phrase.text}")
                print(f"Time: {phrase.offset}")
            
            if user_config["output_file_path"] is not None:
                with open(user_config["output_file_path"], 'w', encoding='utf-8') as f:
                    from json import dumps
                    f.write(dumps(transcription, indent=2))
                print(f"\nTranscription saved to: {user_config['output_file_path']}")


run()
