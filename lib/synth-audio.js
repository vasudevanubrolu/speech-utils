const assert = require('assert');
const fs = require('fs');
const bent = require('bent');
const ttsGoogle = require('@google-cloud/text-to-speech');
const { PollyClient, SynthesizeSpeechCommand } = require('@aws-sdk/client-polly');
const { CartesiaClient } = require('@cartesia/cartesia-js');

const sdk = require('microsoft-cognitiveservices-speech-sdk');
const TextToSpeechV1 = require('ibm-watson/text-to-speech/v1');
const { IamAuthenticator } = require('ibm-watson/auth');
const {
  ResultReason,
  SpeechConfig,
  SpeechSynthesizer,
  CancellationDetails,
  SpeechSynthesisOutputFormat
} = sdk;
const {
  makeSynthKey,
  createNuanceClient,
  createKryptonClient,
  createRivaClient,
  noopLogger,
  makeFilePath,
  makePlayhtKey
} = require('./utils');
const getNuanceAccessToken = require('./get-nuance-access-token');
const getVerbioAccessToken = require('./get-verbio-token');
const {
  SynthesisRequest,
  Voice,
  AudioFormat,
  AudioParameters,
  PCM,
  Input,
  Text,
  SSML,
  EventParameters
} = require('../stubs/nuance/synthesizer_pb');
const {SynthesizeSpeechRequest} = require('../stubs/riva/proto/riva_tts_pb');
const {AudioEncoding} = require('../stubs/riva/proto/riva_audio_pb');
const debug = require('debug')('jambonz:realtimedb-helpers');
const {
  JAMBONES_DISABLE_TTS_STREAMING,
  JAMBONES_DISABLE_AZURE_TTS_STREAMING,
  JAMBONES_HTTP_PROXY_IP,
  JAMBONES_HTTP_PROXY_PORT,
  JAMBONES_TTS_CACHE_DURATION_MINS,
  JAMBONES_TTS_TRIM_SILENCE,
  JAMBONES_AZURE_ENABLE_SSML
} = require('./config');
const EXPIRES = JAMBONES_TTS_CACHE_DURATION_MINS;
const OpenAI = require('openai');
const getAwsAuthToken = require('./get-aws-sts-token');


const trimTrailingSilence = (buffer) => {
  assert.ok(buffer instanceof Buffer, 'trimTrailingSilence - argument is not a Buffer');

  let offset = buffer.length;
  while (offset > 0) {
    // Get 16-bit value from the buffer (read in reverse)
    const value = buffer.readUInt16BE(offset - 2);
    if (value !== 0) {
      break;
    }
    offset -= 2;
  }

  // Trim the silence from the end
  return offset === buffer.length ? buffer : buffer.subarray(0, offset);
};

/**
 * Synthesize speech to an mp3 file, and also cache the generated speech
 * in redis (base64 format) for 24 hours so as to avoid unnecessarily paying
 * time and again for speech synthesis of the same text.
 * It is the responsibility of the caller to unlink the mp3 file after use.
 *
 * @param {*} client - redis client
 * @param {*} logger - pino logger
 * @param {object} opts - options
 * @param {string} opts.vendor - 'google' or 'aws' ('polly' is an alias for 'aws')
 * @param {string} opt.language - language code
 * @param {string} opts.voice - voice identifier
 * @param {string} opts.text - text or ssml to synthesize
 * @param {boolean} opts.disableTtsCache - disable TTS Cache retrieval
 * @returns object containing filepath to an mp3 file in the /tmp folder containing
 * the synthesized audio, and a variable indicating whether it was served from cache
 */
async function synthAudio(client, createHash, retrieveHash, logger, stats, { account_sid,
  vendor, language, voice, gender, text, engine, salt, model, credentials, deploymentId,
  disableTtsCache, renderForCaching = false, disableTtsStreaming, options
}) {
  let audioData;
  let servedFromCache = false;
  let rtt;
  logger = logger || noopLogger;

  assert.ok(['google', 'aws', 'polly', 'microsoft', 'wellsaid', 'nuance', 'nvidia', 'ibm', 'elevenlabs',
    'whisper', 'deepgram', 'playht', 'rimelabs', 'verbio', 'cartesia'].includes(vendor) ||
  vendor.startsWith('custom'),
  `synthAudio supported vendors are google, aws, microsoft, nuance, nvidia and wellsaid ..etc, not ${vendor}`);
  if ('google' === vendor) {
    assert.ok(language, 'synthAudio requires language when google is used');
  }
  else if (['aws', 'polly'].includes(vendor))  {
    assert.ok(voice, 'synthAudio requires voice when aws polly is used');
  }
  else if ('microsoft' === vendor) {
    assert.ok(language || deploymentId, 'synthAudio requires language when microsoft is used');
    assert.ok(voice || deploymentId, 'synthAudio requires voice when microsoft is used');
  }
  else if ('nuance' === vendor) {
    assert.ok(voice, 'synthAudio requires voice when nuance is used');
    if (!credentials.nuance_tts_uri) {
      assert.ok(credentials.client_id, 'synthAudio requires client_id in credentials when nuance is used');
      assert.ok(credentials.secret, 'synthAudio requires client_id in credentials when nuance is used');
    }
  }
  else if ('nvidia' === vendor) {
    assert.ok(voice, 'synthAudio requires voice when nvidia is used');
    assert.ok(language, 'synthAudio requires language when nvidia is used');
    assert.ok(credentials.riva_server_uri, 'synthAudio requires riva_server_uri in credentials when nvidia is used');
  }
  else if ('ibm' === vendor) {
    assert.ok(voice, 'synthAudio requires voice when ibm is used');
    assert.ok(credentials.tts_region, 'synthAudio requires tts_region in credentials when ibm watson is used');
    assert.ok(credentials.tts_api_key, 'synthAudio requires tts_api_key in credentials when nuance is used');
  }
  else if ('wellsaid' === vendor) {
    language = 'en-US'; // WellSaid only supports English atm
    assert.ok(voice, 'synthAudio requires voice when wellsaid is used');
    assert.ok(!text.startsWith('<speak'), 'wellsaid does not support SSML tags');
  } else if ('elevenlabs' === vendor) {
    assert.ok(voice, 'synthAudio requires voice when elevenlabs is used');
    assert.ok(credentials.api_key, 'synthAudio requires api_key when elevenlabs is used');
    assert.ok(credentials.model_id, 'synthAudio requires model_id when elevenlabs is used');
  } else if ('playht' === vendor) {
    assert.ok(voice, 'synthAudio requires voice when playht is used');
    assert.ok(credentials.api_key, 'synthAudio requires api_key when playht is used');
    assert.ok(credentials.user_id, 'synthAudio requires user_id when playht is used');
    assert.ok(credentials.voice_engine, 'synthAudio requires voice_engine when playht is used');
  } else if ('rimelabs' === vendor) {
    assert.ok(voice, 'synthAudio requires voice when rimelabs is used');
    assert.ok(credentials.api_key, 'synthAudio requires api_key when rimelabs is used');
    assert.ok(credentials.model_id, 'synthAudio requires model_id when rimelabs is used');
  } else if ('whisper' === vendor) {
    assert.ok(voice, 'synthAudio requires voice when whisper is used');
    assert.ok(credentials.model_id, 'synthAudio requires model when whisper is used');
    assert.ok(credentials.api_key, 'synthAudio requires api_key when whisper is used');
  } else  if (vendor.startsWith('custom')) {
    assert.ok(credentials.custom_tts_url, `synthAudio requires custom_tts_url in credentials when ${vendor} is used`);
  } else if ('verbio' === vendor) {
    assert.ok(voice, 'synthAudio requires voice when verbio is used');
    assert.ok(credentials.client_id, 'synthAudio requires client_id when verbio is used');
    assert.ok(credentials.client_secret, 'synthAudio requires client_secret when verbio is used');
  } else if ('deepgram' === vendor) {
    if (!credentials.deepgram_tts_uri) {
      assert.ok(credentials.api_key, 'synthAudio requires api_key when deepgram is used');
    }
  } else if ('cartesia' === vendor) {
    assert.ok(credentials.api_key, 'synthAudio requires api_key when cartesia is used');
    assert.ok(credentials.model_id, 'synthAudio requires model_id when cartesia is used');
  }

  const key = makeSynthKey({
    account_sid,
    vendor,
    language: language || '',
    voice: voice || deploymentId,
    engine,
    // model or model_id is used to identify the tts cache.
    model: model || credentials.model_id,
    text
  });

  debug(`synth key is ${key}`);
  let cached;
  if (!disableTtsCache) {
    cached = await client.get(key);
  }
  if (cached) {
    // found in cache - extend the expiry and use it
    debug('result WAS found in cache');
    servedFromCache = true;
    stats.increment('tts.cache.requests', ['found:yes']);
    audioData = JSON.parse(cached);
    // convert base64 audio to buffer
    audioData.audioContent = Buffer.from(audioData.audioContent, 'base64');
    client.expire(key, EXPIRES).catch((err) => logger.info(err, 'Error setting expires'));
  } else {
    // not found in cache - go get it from speech vendor and add to cache
    debug('result was NOT found in cache');
    stats.increment('tts.cache.requests', ['found:no']);
    let vendorLabel = vendor;
    const startAt = process.hrtime();
    switch (vendor) {
      case 'google':
        audioData = await synthGoogle(logger, {credentials, stats, language, voice, gender, text});
        break;
      case 'aws':
      case 'polly':
        vendorLabel = 'aws';
        audioData = await synthPolly(createHash, retrieveHash, logger,
          {credentials, stats, language, voice, text, engine});
        break;
      case 'azure':
      case 'microsoft':
        vendorLabel = 'microsoft';
        audioData = await synthMicrosoft(logger, {credentials, stats, language, voice, text, deploymentId,
          renderForCaching, disableTtsStreaming});
        break;
      case 'nuance':
        model = model || 'enhanced';
        audioData = await synthNuance(client, logger, {credentials, stats, voice, model, text});
        break;
      case 'nvidia':
        audioData = await synthNvidia(client, logger, {credentials, stats, language, voice, model, text,
          renderForCaching, disableTtsStreaming});
        break;
      case 'ibm':
        audioData = await synthIbm(logger, {credentials, stats, voice, text});
        break;
      case 'wellsaid':
        audioData = await synthWellSaid(logger, {credentials, stats, language, voice, text});
        break;
      case 'elevenlabs':
        audioData = await synthElevenlabs(logger, {
          credentials, options, stats, language, voice, text, renderForCaching, disableTtsStreaming});
        break;
      case 'playht':
        audioData = await synthPlayHT(client, logger, {
          credentials, options, stats, language, voice, text, renderForCaching, disableTtsStreaming});
        break;
      case 'cartesia':
        audioData = await synthCartesia(logger, {
          credentials, options, stats, language, voice, text, renderForCaching, disableTtsStreaming});
        break;
      case 'rimelabs':
        audioData = await synthRimelabs(logger, {
          credentials, options, stats, language, voice, text, renderForCaching, disableTtsStreaming});
        break;
      case 'whisper':
        audioData = await synthWhisper(logger, {
          credentials, stats, voice, text, renderForCaching, disableTtsStreaming});
        break;
      case 'verbio':
        audioData = await synthVerbio(client, logger, {
          credentials, stats, voice, text, renderForCaching, disableTtsStreaming});
        if (audioData?.filePath) return audioData;
        break;
      case 'deepgram':
        audioData = await synthDeepgram(logger, {credentials, stats, model, text,
          renderForCaching, disableTtsStreaming});
        break;
      case vendor.startsWith('custom') ? vendor : 'cant_match_value':
        audioData = await synthCustomVendor(logger,
          {credentials, stats, language, voice, text});
        break;
      default:
        assert(`synthAudio: unsupported speech vendor ${vendor}`);
    }
    if ('filePath' in audioData) return audioData;
    const diff = process.hrtime(startAt);
    const time = diff[0] * 1e3 + diff[1] * 1e-6;
    rtt = time.toFixed(0);
    stats.histogram('tts.response_time', rtt, [`vendor:${vendorLabel}`]);
    debug(`tts rtt time for ${text.length} chars on ${vendorLabel}: ${rtt}`);
    logger.info(`tts rtt time for ${text.length} chars on ${vendorLabel}: ${rtt}`);
    // Save audio json to cache
    client.setex(key, EXPIRES, JSON.stringify({
      ...audioData,
      audioContent: audioData.audioContent?.toString('base64')
    }))
      .catch((err) => logger.error(err, `error calling setex on key ${key}`));
  }

  return new Promise((resolve, reject) => {
    const { audioContent, extension } = audioData;
    const filePath = makeFilePath({
      key,
      salt,
      extension
    });
    fs.writeFile(filePath, audioContent, (err) => {
      if (err) return reject(err);
      resolve({filePath, servedFromCache, rtt});
    });
  });
}

const synthPolly = async(createHash, retrieveHash, logger,
  {credentials, stats, language, voice, engine, text}) => {
  try {
    const {region, accessKeyId, secretAccessKey, roleArn} = credentials;
    let polly;
    if (accessKeyId && secretAccessKey) {
      polly = new PollyClient({
        region,
        credentials: {
          accessKeyId,
          secretAccessKey
        }
      });
    } else if (roleArn) {
      polly = new PollyClient({
        region,
        credentials: await getAwsAuthToken(
          logger, createHash, retrieveHash,
          {
            region,
            roleArn
          }),
      });
    } else {
      // AWS RoleArn assigned to Instance profile
      polly = new PollyClient({region});
    }
    const opts = {
      Engine: engine,
      OutputFormat: 'mp3',
      Text: text,
      LanguageCode: language,
      TextType: text.startsWith('<speak>') ? 'ssml' : 'text',
      VoiceId: voice
    };
    const command = new SynthesizeSpeechCommand(opts);
    const data = await polly.send(command);
    const chunks = [];
    return new Promise((resolve, reject) => {
      data.AudioStream
        .on('error', (err) => {
          logger.info({err}, 'synthAudio: Error synthesizing speech using aws polly');
          stats.increment('tts.count', ['vendor:aws', 'accepted:no']);
          reject(err);
        })
        .on('data', (chunk) => {
          chunks.push(chunk);
        })
        .on('end', () => resolve(
          {
            audioContent: Buffer.concat(chunks),
            extension: 'mp3',
            sampleRate: 8000
          }
        ));
    });
  } catch (err) {
    logger.info({err}, 'synthAudio: Error synthesizing speech using aws polly');
    stats.increment('tts.count', ['vendor:aws', 'accepted:no']);
    throw err;
  }
};

const synthGoogle = async(logger, {credentials, stats, language, voice, gender, text}) => {
  const client = new ttsGoogle.TextToSpeechClient(credentials);
  // If google custom voice cloning is used.
  // At this time 31 Oct 2024, google node sdk has not support voice cloning yet.
  if (typeof voice === 'object' && voice.voice_cloning_key) {
    try {
      const accessToken = await client.auth.getAccessToken();
      const projectId = await client.getProjectId();

      const post = bent('https://texttospeech.googleapis.com', 'POST', 'json', {
        'Authorization': `Bearer ${accessToken}`,
        'x-goog-user-project': projectId,
        'Content-Type': 'application/json; charset=utf-8'
      });

      const payload = {
        input: {
          text
        },
        voice: {
          language_code: language,
          voice_clone: {
            voice_cloning_key: voice.voice_cloning_key
          }
        },
        audioConfig: {
          // Cloning voice at this time still in v1 beta version, and it support LINEAR16 in Wav format, 24.000Hz
          audioEncoding: 'LINEAR16',
          sample_rate_hertz: 24000
        }
      };

      const wav = await post('/v1beta1/text:synthesize', payload);
      return {
        audioContent: Buffer.from(wav.audioContent, 'base64'),
        extension: 'wav',
        sampleRate: 24000
      };
    } catch (err) {
      logger.info({err: await err.text()}, 'synthGoogle returned error');
      throw err;
    }
  }

  const opts = {
    voice: {
      ...(typeof voice === 'string' && {name: voice}),
      ...(typeof voice === 'object' && {customVoice: voice}),
      languageCode: language,
      ssmlGender: gender || 'SSML_VOICE_GENDER_UNSPECIFIED'
    },
    audioConfig: {audioEncoding: 'MP3'}
  };
  Object.assign(opts, {input: text.startsWith('<speak>') ? {ssml: text} : {text}});
  try {
    const responses = await client.synthesizeSpeech(opts);
    stats.increment('tts.count', ['vendor:google', 'accepted:yes']);
    client.close();
    return {
      audioContent: responses[0].audioContent,
      extension: 'mp3',
      sampleRate: 8000
    };
  } catch (err) {
    console.error(err);
    logger.info({err, opts}, 'synthAudio: Error synthesizing speech using google');
    stats.increment('tts.count', ['vendor:google', 'accepted:no']);
    client && client.close();
    throw err;
  }
};

const synthIbm = async(logger, {credentials, stats, voice, text}) => {
  const {tts_api_key, tts_region} = credentials;
  const params = {
    text,
    voice,
    accept: 'audio/mp3'
  };

  try {
    const textToSpeech = new TextToSpeechV1({
      authenticator: new IamAuthenticator({
        apikey: tts_api_key,
      }),
      serviceUrl: `https://api.${tts_region}.text-to-speech.watson.cloud.ibm.com`
    });

    const r = await textToSpeech.synthesize(params);
    const chunks = [];
    for await (const chunk of r.result) {
      chunks.push(chunk);
    }
    return {
      audioContent: Buffer.concat(chunks),
      extension: 'mp3',
      sampleRate: 8000
    };
  } catch (err) {
    logger.info({err, params}, 'synthAudio: Error synthesizing speech using ibm');
    stats.increment('tts.count', ['vendor:ibm', 'accepted:no']);
    throw new Error(err.statusText || err.message);
  }
};

async function _synthOnPremMicrosoft(logger, {
  credentials,
  language,
  voice,
  text
}) {
  const {use_custom_tts, custom_tts_endpoint_url, api_key} = credentials;
  let content = text;
  if (use_custom_tts && !content.startsWith('<speak')) {
    /**
     * Note: it seems that to use custom voice ssml is required with the voice attribute
     * Otherwise sending plain text we get "Voice does not match"
     */
    content = `<speak>${text}</speak>`;
  }

  if (content.startsWith('<speak>')) {
    /* microsoft enforces some properties and uses voice xml element so if the user did not supply do it for them */
    const words = content.slice(7, -8).trim().replace(/(\r\n|\n|\r)/gm, ' ');
    // eslint-disable-next-line max-len
    content = `<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="${language}"><voice name="${voice}">${words}</voice></speak>`;
    logger.info({content}, 'synthMicrosoft');
  }

  try {
    const trimSilence = JAMBONES_TTS_TRIM_SILENCE;
    const post = bent('POST', 'buffer', {
      'X-Microsoft-OutputFormat': trimSilence ? 'raw-8khz-16bit-mono-pcm' : 'audio-16khz-32kbitrate-mono-mp3',
      'Content-Type': 'application/ssml+xml',
      'User-Agent': 'Jambonz',
      ...(api_key && {'Ocp-Apim-Subscription-Key': api_key})
    });
    const audioContent = await post(custom_tts_endpoint_url, content);
    return {
      audioContent,
      extension: trimSilence ? 'r8' : 'mp3',
      sampleRate: 8000
    };
  } catch (err) {
    logger.info({err}, '_synthMicrosoftByHttp returned error');
    throw err;
  }
}

const synthMicrosoft = async(logger, {
  credentials,
  stats,
  language,
  voice,
  text,
  renderForCaching,
  disableTtsStreaming
}) => {
  try {
    const {api_key: apiKey, region, use_custom_tts, custom_tts_endpoint, custom_tts_endpoint_url} = credentials;
    // let clean up the text
    let content = text;
    if (use_custom_tts && !content.startsWith('<speak')) {
      /**
     * Note: it seems that to use custom voice ssml is required with the voice attribute
     * Otherwise sending plain text we get "Voice does not match"
     */
      content = `<speak>${text}</speak>`;
    }

    if (content.startsWith('<speak>')) {
    /* microsoft enforces some properties and uses voice xml element so if the user did not supply do it for them */
      const words = content.slice(7, -8).trim().replace(/(\r\n|\n|\r)/gm, ' ');
      // eslint-disable-next-line max-len
      content = `<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="${language}"><voice name="${voice}">${words}</voice></speak>`;
      logger.info({content}, 'synthMicrosoft');
    }
    if (JAMBONES_AZURE_ENABLE_SSML) {
      // eslint-disable-next-line max-len
      content = `<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="${language}"><voice name="${voice}">${text}</voice></speak>`;
    }
    if (!JAMBONES_DISABLE_TTS_STREAMING && !JAMBONES_DISABLE_AZURE_TTS_STREAMING &&
      !renderForCaching && !disableTtsStreaming) {
      let params = '';
      params += `{api_key=${apiKey}`;
      params += `,language=${language}`;
      params += ',vendor=microsoft';
      params += `,voice=${voice}`;
      params += ',write_cache_file=1';
      if (region) params += `,region=${region}`;
      if (custom_tts_endpoint) params += `,endpointId=${custom_tts_endpoint}`;
      if (custom_tts_endpoint_url) params += `,endpoint=${custom_tts_endpoint_url}`;
      if (JAMBONES_HTTP_PROXY_IP) params += `,http_proxy_ip=${JAMBONES_HTTP_PROXY_IP}`;
      if (JAMBONES_HTTP_PROXY_PORT) params += `,http_proxy_port=${JAMBONES_HTTP_PROXY_PORT}`;
      params += '}';
      return {
        filePath: `say:${params}${content.replace(/\n/g, ' ')}`,
        servedFromCache: false,
        rtt: 0
      };
    }
    // Azure Onprem
    if (use_custom_tts && custom_tts_endpoint_url) {
      return await _synthOnPremMicrosoft(logger, {
        credentials,
        stats,
        language,
        voice,
        text
      });
    }
    // Azure hosted service
    const trimSilence = JAMBONES_TTS_TRIM_SILENCE;
    const speechConfig = SpeechConfig.fromSubscription(apiKey, region);
    speechConfig.speechSynthesisLanguage = language;
    speechConfig.speechSynthesisVoiceName = voice;
    if (use_custom_tts && custom_tts_endpoint) {
      speechConfig.endpointId = custom_tts_endpoint;
    }
    speechConfig.speechSynthesisOutputFormat = trimSilence ?
      SpeechSynthesisOutputFormat.Raw8Khz16BitMonoPcm :
      SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3;

    if (JAMBONES_HTTP_PROXY_IP && JAMBONES_HTTP_PROXY_PORT) {
      logger.debug(
        `synthMicrosoft: using proxy ${JAMBONES_HTTP_PROXY_IP}:${JAMBONES_HTTP_PROXY_PORT}`);
      speechConfig.setProxy(JAMBONES_HTTP_PROXY_IP, JAMBONES_HTTP_PROXY_PORT);
    }
    const synthesizer = new SpeechSynthesizer(speechConfig);

    return new Promise((resolve, reject) => {
      const speakAsync = content.startsWith('<speak') ?
        synthesizer.speakSsmlAsync.bind(synthesizer) :
        synthesizer.speakTextAsync.bind(synthesizer);
      speakAsync(
        content,
        async(result) => {
          switch (result.reason) {
            case ResultReason.Canceled:
              const cancellation = CancellationDetails.fromResult(result);
              logger.info({reason: cancellation.errorDetails}, 'synthAudio: (Microsoft) synthesis canceled');
              synthesizer.close();
              reject(cancellation.errorDetails);
              break;
            case ResultReason.SynthesizingAudioCompleted:
              let buffer = Buffer.from(result.audioData);
              if (trimSilence) buffer = trimTrailingSilence(buffer);
              resolve({
                audioContent: buffer,
                extension: trimSilence ? 'r8' : 'mp3',
                sampleRate: 8000
              });
              synthesizer.close();
              stats.increment('tts.count', ['vendor:microsoft', 'accepted:yes']);
              break;
            default:
              logger.info({result}, 'synthAudio: (Microsoft) unexpected result');
              break;
          }
        },
        (err) => {
          logger.info({err}, 'synthAudio: (Microsoft) error synthesizing');
          stats.increment('tts.count', ['vendor:microsoft', 'accepted:no']);
          synthesizer.close();
          reject(err);
        });
    });
  } catch (err) {
    logger.info({err}, 'synthAudio: Error synthesizing speech using Microsoft');
    stats.increment('tts.count', ['vendor:google', 'accepted:no']);
  }
};

const synthWellSaid = async(logger, {credentials, stats, language, voice, gender, text}) => {
  const {api_key} = credentials;
  try {
    const post = bent('https://api.wellsaidlabs.com', 'POST', 'buffer', {
      'X-Api-Key': api_key,
      'Accept': 'audio/mpeg',
      'Content-Type': 'application/json'
    });
    const audioContent = await post('/v1/tts/stream', {
      text,
      speaker_id: voice
    });
    return {
      audioContent,
      extension: 'mp3',
      sampleRate: 8000
    };
  } catch (err) {
    logger.info({err}, 'testWellSaidTts returned error');
    throw err;
  }
};

const synthNuance = async(client, logger, {credentials, stats, voice, model, text}) => {
  let nuanceClient;
  const {client_id, secret, nuance_tts_uri} = credentials;
  if (nuance_tts_uri) {
    nuanceClient = await createKryptonClient(nuance_tts_uri);
  }
  else {
    /* get a nuance access token */
    const {access_token} = await getNuanceAccessToken(client, logger, client_id, secret, 'tts');
    nuanceClient = await createNuanceClient(access_token);
  }

  const v = new Voice();
  const p = new AudioParameters();
  const f = new AudioFormat();
  const pcm = new PCM();
  const params  = new EventParameters();
  const request = new SynthesisRequest();
  const input = new Input();

  if (text.startsWith('<speak')) {
    const ssml = new SSML();
    ssml.setText(text);
    input.setSsml(ssml);
  }
  else {
    const t = new Text();
    t.setText(text);
    input.setText(t);
  }
  const sampleRate = 8000;
  pcm.setSampleRateHz(sampleRate);
  f.setPcm(pcm);
  p.setAudioFormat(f);
  v.setName(voice);
  v.setModel(model);
  request.setVoice(v);
  request.setAudioParams(p);
  request.setInput(input);
  request.setEventParams(params);
  request.setUserId('jambonz');

  return new Promise((resolve, reject) => {
    nuanceClient.unarySynthesize(request, (err, response) => {
      if (err) {
        console.error(err);
        return reject(err);
      }
      const status = response.getStatus();
      const code = status.getCode();
      if (code !== 200) {
        const message = status.getMessage();
        const details = status.getDetails();
        return reject({code, message, details});
      }
      resolve({
        audioContent: Buffer.from(response.getAudio()),
        extension: 'r8',
        sampleRate
      });
    });
  });
};

const synthNvidia = async(client, logger, {
  credentials, stats, language,  voice, model, text, renderForCaching, disableTtsStreaming
}) => {
  const {riva_server_uri} = credentials;
  if (!JAMBONES_DISABLE_TTS_STREAMING && !renderForCaching && !disableTtsStreaming) {
    let params = '';
    params += `{riva_server_uri=${riva_server_uri}`;
    params += `,voice=${voice}`;
    params += `,language=${language}`;
    params += ',write_cache_file=1';
    params += '}';

    return {
      filePath: `say:${params}${text.replace(/\n/g, ' ')}`,
      servedFromCache: false,
      rtt: 0
    };
  }
  let rivaClient, request;
  const sampleRate = 8000;
  try {
    rivaClient = await createRivaClient(riva_server_uri);
    request = new SynthesizeSpeechRequest();
    request.setVoiceName(voice);
    request.setLanguageCode(language);
    request.setSampleRateHz(sampleRate);
    request.setEncoding(AudioEncoding.LINEAR_PCM);
    request.setText(text);
  } catch (err) {
    logger.info({err}, 'error creating riva client');
    return Promise.reject(err);
  }

  return new Promise((resolve, reject) => {
    rivaClient.synthesize(request, (err, response) => {
      if (err) {
        logger.info({err, voice, language}, 'error synthesizing speech using Nvidia');
        return reject(err);
      }
      resolve({
        audioContent: Buffer.from(response.getAudio()),
        extension: 'r8',
        sampleRate
      });
    });
  });
};


const synthCustomVendor = async(logger, {credentials, stats, language, voice, text, filePath}) => {
  const {vendor, auth_token, custom_tts_url} = credentials;

  try {
    const post = bent('POST', {
      'Authorization': `Bearer ${auth_token}`,
      'Content-Type': 'application/json'
    });

    const response = await post(custom_tts_url, {
      language,
      voice,
      type: text.startsWith('<speak>') ? 'ssml' : 'text',
      text
    });

    const mime = response.headers['content-type'];
    const buffer = await response.arrayBuffer();
    const [extension, sampleRate] = getFileExtFromMime(mime);
    return {
      audioContent: buffer,
      extension,
      sampleRate
    };
  } catch (err) {
    logger.info({err}, `Vendor ${vendor} returned error`);
    throw err;
  }
};

const synthElevenlabs = async(logger, {
  credentials, options, stats, voice, text, renderForCaching, disableTtsStreaming
}) => {
  const {api_key, model_id, options: credOpts} = credentials;
  const opts = !!options && Object.keys(options).length !== 0 ? options : JSON.parse(credOpts || '{}');

  /* default to using the streaming interface, unless disabled by env var OR we want just a cache file */
  if (!JAMBONES_DISABLE_TTS_STREAMING && !renderForCaching && !disableTtsStreaming) {
    let params = '';
    params += `{api_key=${api_key}`;
    params += ',vendor=elevenlabs';
    params += `,voice=${voice}`;
    params += `,model_id=${model_id}`;
    params += `,optimize_streaming_latency=${opts.optimize_streaming_latency || 2}`;
    params += ',write_cache_file=1';
    if (opts.voice_settings?.similarity_boost) params += `,similarity_boost=${opts.voice_settings.similarity_boost}`;
    if (opts.voice_settings?.stability) params += `,stability=${opts.voice_settings.stability}`;
    if (opts.voice_settings?.style) params += `,style=${opts.voice_settings.style}`;
    if (opts.voice_settings?.speed !== null && opts.voice_settings?.speed !== undefined)
      params += `,speed=${opts.voice_settings.speed}`;
    if (opts.voice_settings?.use_speaker_boost === false) params += ',use_speaker_boost=false';
    if (opts.previous_text) params += `,previous_text=${opts.previous_text}`;
    if (opts.next_text) params += `,next_text=${opts.next_text}`;
    if (opts.pronunciation_dictionary_locators && Array.isArray(opts.pronunciation_dictionary_locators))
      params += `,pronunciation_dictionary_locators=${JSON.stringify(opts.pronunciation_dictionary_locators)}`;
    params += '}';

    return {
      filePath: `say:${params}${text.replace(/\n/g, ' ').replace(/\r/g, ' ')}`,
      servedFromCache: false,
      rtt: 0
    };
  }

  const optimize_streaming_latency = opts.optimize_streaming_latency ?
    `?optimize_streaming_latency=${opts.optimize_streaming_latency}` : '';
  try {
    const post = bent('https://api.elevenlabs.io', 'POST', 'buffer', {
      'xi-api-key': api_key,
      'Accept': 'audio/mpeg',
      'Content-Type': 'application/json'
    });
    const audioContent = await post(`/v1/text-to-speech/${voice}${optimize_streaming_latency}`, {
      text,
      model_id,
      voice_settings: {
        stability: 0.5,
        similarity_boost: 0.5
      },
      ...opts
    });
    return {
      audioContent,
      extension: 'mp3',
      sampleRate: 8000
    };
  } catch (err) {
    logger.info({err}, 'synth Elevenlabs returned error');
    stats.increment('tts.count', ['vendor:elevenlabs', 'accepted:no']);
    throw err;
  }
};

const synthPlayHT = async(client, logger, {
  credentials, options, stats, voice, language, text, renderForCaching, disableTtsStreaming
}) => {
  const {api_key, user_id, voice_engine, options: credOpts} = credentials;
  const opts = !!options && Object.keys(options).length !== 0 ? options : JSON.parse(credOpts || '{}');

  let synthesizeUrl = 'https://api.play.ht/api/v2/tts/stream';

  // If model is play3.0, the synthesizeUrl is got from authentication endpoint
  if (voice_engine === 'Play3.0') {
    try {
      const post = bent('https://api.play.ht', 'POST', 'json', 201, {
        'AUTHORIZATION': api_key,
        'X-USER-ID': user_id,
        'Accept': 'application/json'
      });
      const key = makePlayhtKey(api_key);
      const url = await client.get(key);
      if (!url) {
        const {inference_address, expires_at_ms} = await post('/api/v3/auth');
        synthesizeUrl = inference_address;
        const expiry =  Math.floor((expires_at_ms - Date.now()) / 1000 - 30);
        await client.set(key, inference_address, 'EX', expiry);
      } else {
        // Use cached URL
        synthesizeUrl = url;
      }
    } catch (err) {
      logger.info({err}, 'synth PlayHT returned error for authentication version 3.0');
      stats.increment('tts.count', ['vendor:playht', 'accepted:no']);
      throw err;
    }
  }

  /* default to using the streaming interface, unless disabled by env var OR we want just a cache file */
  if (!JAMBONES_DISABLE_TTS_STREAMING && !renderForCaching && !disableTtsStreaming) {
    let params = '';
    params += `{api_key=${api_key}`;
    params += `,user_id=${user_id}`;
    params += ',vendor=playht';
    params += `,voice=${voice}`;
    params += `,voice_engine=${voice_engine}`;
    params += `,synthesize_url=${synthesizeUrl}`;
    params += ',write_cache_file=1';
    params += `,language=${language}`;
    if (opts.quality) params += `,quality=${opts.quality}`;
    if (opts.speed) params += `,speed=${opts.speed}`;
    if (opts.seed) params += `,style=${opts.seed}`;
    if (opts.temperature) params += `,temperature=${opts.temperature}`;
    if (opts.emotion) params += `,emotion=${opts.emotion}`;
    if (opts.voice_guidance) params += `,voice_guidance=${opts.voice_guidance}`;
    if (opts.style_guidance) params += `,style_guidance=${opts.style_guidance}`;
    if (opts.text_guidance) params += `,text_guidance=${opts.text_guidance}`;
    if (opts.top_p) params += `,top_p=${opts.top_p}`;
    if (opts.repetition_penalty) params += `,repetition_penalty=${opts.repetition_penalty}`;
    params += '}';

    return {
      filePath: `say:${params}${text.replace(/\n/g, ' ').replace(/\r/g, ' ')}`,
      servedFromCache: false,
      rtt: 0
    };
  }

  try {
    const post = bent('POST', 'buffer', {
      ...(voice_engine !== 'Play3.0' && {
        'AUTHORIZATION': api_key,
        'X-USER-ID': user_id,
      }),
      'Accept': 'audio/mpeg',
      'Content-Type': 'application/json'
    });

    const audioContent = await post(synthesizeUrl, {
      text,
      ...(voice_engine === 'Play3.0' && { language }),
      voice,
      voice_engine,
      output_format: 'mp3',
      sample_rate: 8000,
      ...opts
    });
    return {
      audioContent,
      extension: 'mp3',
      sampleRate: 8000
    };
  } catch (err) {
    logger.info({err}, 'synth PlayHT returned error');
    stats.increment('tts.count', ['vendor:playht', 'accepted:no']);
    throw err;
  }
};

const synthRimelabs = async(logger, {
  credentials, options, stats, language, voice, text, renderForCaching, disableTtsStreaming
}) => {
  const {api_key, model_id, options: credOpts} = credentials;
  const opts = !!options && Object.keys(options).length !== 0 ? options : JSON.parse(credOpts || '{}');

  /* default to using the streaming interface, unless disabled by env var OR we want just a cache file */
  if (!JAMBONES_DISABLE_TTS_STREAMING && !renderForCaching && !disableTtsStreaming) {
    let params = '';
    params += `{api_key=${api_key}`;
    params += `,model_id=${model_id}`;
    params += ',vendor=rimelabs';
    params += `,language=${language}`;
    params += `,voice=${voice}`;
    params += ',write_cache_file=1';
    if (opts.speedAlpha) params += `,speed_alpha=${opts.speedAlpha}`;
    if (opts.reduceLatency) params += `,reduce_latency=${opts.reduceLatency}`;
    // Arcana model parameters
    if (opts.temperature) params += `,temperature=${opts.temperature}`;
    if (opts.repetition_penalty) params += `,repetition_penalty=${opts.repetition_penalty}`;
    if (opts.top_p) params += `,top_p=${opts.top_p}`;
    if (opts.max_tokens) params += `,max_tokens=${opts.max_tokens}`;
    params += '}';

    return {
      filePath: `say:${params}${text.replace(/\n/g, ' ').replace(/\r/g, ' ')}`,
      servedFromCache: false,
      rtt: 0
    };
  }

  try {
    const post = bent('https://users.rime.ai', 'POST', 'buffer', {
      'Authorization': `Bearer ${api_key}`,
      'Accept': 'audio/mp3',
      'Content-Type': 'application/json'
    });
    const sampleRate = 8000;
    const audioContent = await post('/v1/rime-tts', {
      speaker: voice,
      text,
      modelId: model_id,
      samplingRate: sampleRate,
      lang: language,
      ...opts
    });
    return {
      audioContent,
      extension: 'mp3',
      sampleRate
    };
  } catch (err) {
    logger.info({err}, 'synth rimelabs returned error');
    stats.increment('tts.count', ['vendor:rimelabs', 'accepted:no']);
    throw err;
  }
};
const synthVerbio = async(client, logger, {credentials, stats, voice, text, renderForCaching, disableTtsStreaming}) => {
  //https://doc.speechcenter.verbio.com/#tag/Text-To-Speech-REST-API
  if (text.length > 2000) {
    throw new Error('Verbio cannot synthesize for the text length larger than 2000 characters');
  }
  const token = await getVerbioAccessToken(client, logger, credentials);
  if (!process.env.JAMBONES_DISABLE_TTS_STREAMING && !renderForCaching && !disableTtsStreaming) {
    let params = '';
    params += `{access_token=${token.access_token}`;
    params += ',vendor=verbio';
    params += `,voice=${voice}`;
    params += ',write_cache_file=1';
    params += '}';

    return {
      filePath: `say:${params}${text.replace(/\n/g, ' ')}`,
      servedFromCache: false,
      rtt: 0
    };
  }

  try {
    const post = bent('https://us.rest.speechcenter.verbio.com', 'POST', 'buffer', {
      'Authorization': `Bearer ${token.access_token}`,
      'User-Agent': 'jambonz',
      'Content-Type': 'application/json'
    });
    const audioContent = await post('/api/v1/synthesize', {
      voice_id: voice,
      output_sample_rate: '8k',
      output_encoding: 'pcm16',
      text
    });
    return {
      audioContent,
      extension: 'r8',
      sampleRate: 8000
    };
  } catch (err) {
    logger.info({err}, 'synth Verbio returned error');
    stats.increment('tts.count', ['vendor:verbio', 'accepted:no']);
    throw err;
  }
};

const synthWhisper = async(logger, {credentials, stats, voice, text, renderForCaching, disableTtsStreaming}) => {
  const {api_key, model_id, baseURL, timeout, speed} = credentials;
  /* if the env is set to stream then bag out, unless we are specifically rendering to generate a cache file */
  if (!JAMBONES_DISABLE_TTS_STREAMING && !renderForCaching && !disableTtsStreaming) {
    let params = '';
    params += `{api_key=${api_key}`;
    params += `,model_id=${model_id}`;
    params += ',vendor=whisper';
    params += `,voice=${voice}`;
    params += ',write_cache_file=1';
    if (speed) params += `,speed=${speed}`;
    params += '}';

    return {
      filePath: `say:${params}${text.replace(/\n/g, ' ')}`,
      servedFromCache: false,
      rtt: 0
    };
  }
  try {
    const openai = new OpenAI.OpenAI({
      apiKey: api_key,
      timeout: timeout || 5000,
      ...(baseURL && {baseURL})
    });

    const mp3 = await openai.audio.speech.create({
      model: model_id,
      voice,
      input: text,
      response_format: 'mp3'
    });
    return {
      audioContent: Buffer.from(await mp3.arrayBuffer()),
      extension: 'mp3',
      sampleRate: 8000
    };
  } catch (err) {
    logger.info({err}, 'synth whisper returned error');
    stats.increment('tts.count', ['vendor:openai', 'accepted:no']);
    throw err;
  }
};

const synthDeepgram = async(logger, {credentials, stats, model, text, renderForCaching, disableTtsStreaming}) => {
  const {api_key, deepgram_tts_uri} = credentials;
  if (!JAMBONES_DISABLE_TTS_STREAMING && !renderForCaching && !disableTtsStreaming) {
    let params = '';
    params += `{api_key=${api_key}`;
    params += ',vendor=deepgram';
    params += `,voice=${model}`;
    params += ',write_cache_file=1';
    if (deepgram_tts_uri) params += `,endpoint=${deepgram_tts_uri}`;
    params += '}';

    return {
      filePath: `say:${params}${text.replace(/\n/g, ' ')}`,
      servedFromCache: false,
      rtt: 0
    };
  }
  try {
    const post = bent(deepgram_tts_uri || 'https://api.deepgram.com', 'POST', 'buffer', {
      // on-premise deepgram does not  require to have api_key
      ...(api_key && {'Authorization': `Token ${api_key}`}),
      'Accept': 'audio/mpeg',
      'Content-Type': 'application/json'
    });
    const audioContent = await post(`/v1/speak?model=${model}`, {
      text
    });
    return {
      audioContent,
      extension: 'mp3',
      sampleRate: 8000
    };
  } catch (err) {
    logger.info({err}, 'synth Deepgram returned error');
    stats.increment('tts.count', ['vendor:deepgram', 'accepted:no']);
    throw err;
  }
};

const synthCartesia = async(logger, {
  credentials, options, stats, voice, language, text, renderForCaching, disableTtsStreaming
}) => {
  const {api_key, model_id, embedding, options: credOpts} = credentials;
  const opts = !!options && Object.keys(options).length !== 0 ? options : JSON.parse(credOpts || '{}');

  if (!JAMBONES_DISABLE_TTS_STREAMING && !renderForCaching && !disableTtsStreaming) {
    let params = '';
    params += `{api_key=${api_key}`;
    params += `,model_id=${model_id}`;
    params += ',vendor=cartesia';
    params += `,voice=${voice}`;
    params += ',write_cache_file=1';
    params += `,language=${language}`;
    params += `,voice_mode=${embedding ? 'embedding' : 'id'}`;
    if (embedding) params += `,embedding=${embedding}`;
    if (opts.speed) params += `,speed=${opts.speed}`;
    if (opts.emotion) params += `,emotion=${opts.emotion}`;
    params += '}';

    return {
      filePath: `say:${params}${text.replace(/\n/g, ' ').replace(/\r/g, ' ')}`,
      servedFromCache: false,
      rtt: 0
    };
  }

  try {
    const client = new CartesiaClient({ apiKey: api_key });
    const sampleRate = 48000;
    const mp3 = await client.tts.bytes({
      modelId: model_id,
      transcript: text,
      voice: {
        mode: embedding ? 'embedding' : 'id',
        ...(embedding ?
          {
            embedding: embedding.split(',').map(Number)
          } :
          {
            id: voice
          }
        ),
        ...(opts.speed || opts.emotion && {
          experimentalControls: {
            ...(opts.speed !== null && opts.speed !== undefined && {speed: opts.speed}),
            ...(opts.emotion && {emotion: opts.emotion}),
          }
        })
      },
      language: language,
      outputFormat: {
        container: 'mp3',
        bitRate: 128000,
        sampleRate
      },
    });
    return {
      audioContent: Buffer.from(mp3),
      extension: 'mp3',
      sampleRate
    };
  } catch (err) {
    logger.info({err}, 'synth Cartesia returned error');
    stats.increment('tts.count', ['vendor:cartesia', 'accepted:no']);
    throw err;
  }

};

const getFileExtFromMime = (mime) => {
  switch (mime) {
    case 'audio/wav':
    case 'audio/x-wav':
      return ['wav', 8000];
    case /audio\/l16.*rate=8000/.test(mime) ? mime : 'cant match value':
      return ['r8', 8000];
    case /audio\/l16.*rate=16000/.test(mime) ? mime : 'cant match value':
      return ['r16', 16000];
    case /audio\/l16.*rate=24000/.test(mime) ? mime : 'cant match value':
      return ['r24', 24000];
    case /audio\/l16.*rate=32000/.test(mime) ? mime : 'cant match value':
      return ['r32', 32000];
    case /audio\/l16.*rate=48000/.test(mime) ? mime : 'cant match value':
      return ['r48', 48000];
    case 'audio/mpeg':
    case 'audio/mp3':
      return ['mp3', 8000];
    default:
      return ['wav', 8000];
  }
};

module.exports = synthAudio;
