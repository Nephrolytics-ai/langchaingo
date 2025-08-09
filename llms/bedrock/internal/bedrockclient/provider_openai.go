package bedrockclient

import (
	"context"
	"encoding/json"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/tmc/langchaingo/llms"
)

// Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html
type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// metaTextGenerationInput is the input to the model.
type openaiTextGenerationInput struct {
	// The prompt that you want to pass to the model. Required
	Messages []openAIMessage `json:"messages"`
	// Used to control the randomness of the generation. Optional, default = 0.5
	Temperature float64 `json:"temperature,omitempty"`
	// Used to lower value to ignore less probable options. Optional, default = 0.9
	TopP float64 `json:"top_p,omitempty"`
}

// metaTextGenerationOutput is the output from the model.
type openaiTextGenerationOutput struct {
	Id      string `json:"id"`
	Choices []struct {
		FinishReason string `json:"finish_reason"`
		Index        int    `json:"index"`
		Message      struct {
			Content string `json:"content"`
			Role    string `json:"role"`
		} `json:"message"`
	} `json:"choices"`
	Created           int    `json:"created"`
	Model             string `json:"model"`
	ServiceTier       string `json:"service_tier"`
	SystemFingerprint string `json:"system_fingerprint"`
	Object            string `json:"object"`
	Usage             struct {
		CompletionTokens int `json:"completion_tokens"`
		PromptTokens     int `json:"prompt_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// Finish reason for the completion of the generation.
const (
	OpenAICompletionReasonStop   = "stop"
	OpenAICompletionReasonLength = "length"
)

func createOpenAICompletion(ctx context.Context,
	client *bedrockruntime.Client,
	modelID string,
	messages []Message,
	options llms.CallOptions,
) (*llms.ContentResponse, error) {
	//txt := processInputMessagesGeneric(messages)
	var openAIMessages []openAIMessage
	for _, msg := range messages {
		role := "developer"
		if msg.Role != "system" {
			role = "system"
		}
		x := openAIMessage{
			Role:    role,
			Content: msg.Content,
		}
		openAIMessages = append(openAIMessages, x)
	}

	input := &openaiTextGenerationInput{
		Messages:    openAIMessages,
		Temperature: options.Temperature,
		TopP:        options.TopP,
	}

	body, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}

	modelInput := &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelID),
		Accept:      aws.String("*/*"),
		ContentType: aws.String("application/json"),
		Body:        body,
	}

	resp, err := client.InvokeModel(ctx, modelInput)
	if err != nil {
		return nil, err
	}

	var output openaiTextGenerationOutput

	err = json.Unmarshal(resp.Body, &output)
	if err != nil {
		return nil, err
	}

	return &llms.ContentResponse{
		Choices: []*llms.ContentChoice{
			{
				Content:    output.Choices[0].Message.Content,
				StopReason: output.Choices[0].FinishReason,
				GenerationInfo: map[string]interface{}{
					"input_tokens":  output.Usage.PromptTokens,
					"output_tokens": output.Usage.CompletionTokens,
				},
			},
		},
	}, nil
}
