import random
import pytest
import torch
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from zeroband.models.llama import Transformer, llama2_configs
from zeroband.models.llama.model import Attention, ModelArgs, create_block_mask_from_seqlens
from zeroband.models.hf_llama import load_llama_model
from zeroband.models.hf_gpt2 import load_gpt2_model
from zeroband.config import Config, DataConfig, TrainConfig, OptimConfig


VOCAB_SIZE = 1024

ERROR_ATOL = {
    torch.float: 3e-4,
    torch.half: 4e-3,
    torch.bfloat16: 2e-2,
}
ERROR_RTOL = {
    torch.float: 2e-5,
    torch.half: 4e-4,
    torch.bfloat16: 5e-3,
}


@pytest.fixture
def llama_config() -> ModelArgs:
    config = llama2_configs["debugmodel"]
    config.vocab_size = VOCAB_SIZE
    return config


def test_llama(llama_config: ModelArgs):
    seq_len = 512
    bs = 8
    model = Transformer(llama_config).to("cuda")
    input_ = torch.randint(0, llama_config.vocab_size, (bs, seq_len)).to("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_)

    assert output.shape == (bs, seq_len, llama_config.vocab_size)


def get_freqs_cis(llama_config: ModelArgs):
    model = Transformer(llama_config).to("cuda")
    return model.freqs_cis


def test_attn(llama_config: ModelArgs):
    seq_len = 512
    bs = 8

    freqs_cis = get_freqs_cis(llama_config)
    input_ = torch.rand(bs, seq_len, llama_config.dim).to("cuda")
    seqlens = [torch.Tensor([seq_len]).int().to("cuda") for _ in range(bs)]
    block_mask = create_block_mask_from_seqlens(seqlens)

    attn = Attention(llama_config).to("cuda")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output_sdpa = attn(input_, freqs_cis)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output_flex = attn(input_, freqs_cis, block_mask=block_mask)

    rtol = ERROR_RTOL[torch.bfloat16]
    atol = ERROR_ATOL[torch.bfloat16]
    assert output_sdpa.shape == output_flex.shape
    torch.testing.assert_close(output_sdpa, output_flex, rtol=rtol, atol=atol)


def test_packing_simple(llama_config: ModelArgs):
    seq_len = 512
    bs = 8

    freqs_cis = get_freqs_cis(llama_config)
    input_ = torch.rand(bs, seq_len, llama_config.dim).to("cuda")
    seqlens = [torch.Tensor([seq_len // 4] * 4).int().to("cuda") for _ in range(bs)]
    block_mask = create_block_mask_from_seqlens(seqlens)

    attn = Attention(llama_config).to("cuda")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = attn(input_, freqs_cis, block_mask=block_mask)

    assert output.shape == (bs, seq_len, llama_config.dim)


def test_sequence_packing_two_time_same_sequence(llama_config: ModelArgs):
    """
    In this test we take a sequence and pack it with itself along the seqlen dimension.
    We then pass the packed sequence to the attention layer and check that the output for each sequence is the same.
    """

    model = Attention(llama_config).to("cuda")

    emb = torch.nn.Embedding(10, llama_config.dim).to("cuda")

    seq = [2, 1, 4, 8]
    input_stuff_raw = torch.Tensor([seq + seq]).long().to("cuda")
    seqlens = [torch.Tensor([len(seq), len(seq)]).int().to("cuda")]
    block_mask = create_block_mask_from_seqlens(seqlens)

    input_stuff = emb(input_stuff_raw)

    freqs_cis = get_freqs_cis(llama_config)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_stuff, freqs_cis, block_mask=block_mask)

    output_left = output[:, :4, :]
    output_right = output[:, 4:, :]

    ### TESTING
    assert output_left.shape == output_right.shape

    rtol = ERROR_RTOL[torch.bfloat16]
    atol = ERROR_ATOL[torch.bfloat16]
    torch.testing.assert_close(output_left, output_right, atol=atol, rtol=rtol)


def test_sequence_packing_vs_normal(llama_config: ModelArgs):
    """
    take two sequences and compare the outout of attention on individual sequences vs the output of attention on the packed sequence
    """

    model = Attention(llama_config).to("cuda")
    emb = torch.nn.Embedding(10, llama_config.dim).to("cuda")

    freqs_cis = get_freqs_cis(llama_config)

    seq_1 = [2, 1, 4, 8]
    seq_2 = [3, 7, 5, 6]

    input_packed_raw = torch.Tensor([seq_1 + seq_2]).long().to("cuda")
    seqlens = [torch.Tensor([len(seq_1), len(seq_2)]).int().to("cuda")]
    block_mask = create_block_mask_from_seqlens(seqlens)

    input_packed = emb(input_packed_raw)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_packed, freqs_cis, block_mask=block_mask)

    output_packed_1 = output[:, :4, :]
    output_packed_2 = output[:, 4:, :]

    input_raw_1 = torch.Tensor([seq_1]).long().to("cuda")
    input_raw_2 = torch.Tensor([seq_2]).long().to("cuda")

    emb_1 = emb(input_raw_1)
    emb_2 = emb(input_raw_2)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output_1 = model(emb_1, freqs_cis)
        output_2 = model(emb_2, freqs_cis)

    rtol = ERROR_RTOL[torch.bfloat16]
    atol = ERROR_ATOL[torch.bfloat16]

    ### TESTING
    assert output_1.shape == output_packed_1.shape
    assert output_2.shape == output_packed_2.shape

    torch.testing.assert_close(output_1, output_packed_1, atol=atol, rtol=rtol)
    torch.testing.assert_close(output_2, output_packed_2, atol=atol, rtol=rtol)


def test_sequence_packing_vs_normal_random(llama_config: ModelArgs):
    """
    take two sequences and compare the outout of attention on individual sequences vs the output of attention on the packed sequence
    """

    model = Attention(llama_config).to("cuda")

    freqs_cis = get_freqs_cis(llama_config)

    MAX_SEQ_LEN = 256

    for _ in range(10):
        seq_len_cutoff = random.randint(1, MAX_SEQ_LEN)

        seq1 = seq_len_cutoff
        seq2 = MAX_SEQ_LEN - seq_len_cutoff
        input_1 = torch.rand(1, seq1, llama_config.dim).to("cuda")
        input_2 = torch.rand(1, seq2, llama_config.dim).to("cuda")

        seqlens = [torch.Tensor([seq1, seq2]).int().to("cuda")]
        block_mask = create_block_mask_from_seqlens(seqlens)

        packed_input = torch.cat([input_1, input_2], dim=1)

        # packed output
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(packed_input, freqs_cis, block_mask=block_mask)

        output_packed_1 = output[:, :seq_len_cutoff, :]
        output_packed_2 = output[:, seq_len_cutoff:, :]

        # normal output
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output_1 = model(input_1, freqs_cis)
            output_2 = model(input_2, freqs_cis)

        rtol = ERROR_RTOL[torch.bfloat16]
        atol = ERROR_ATOL[torch.bfloat16]

        ### TESTING
        assert output_1.shape == output_packed_1.shape
        assert output_2.shape == output_packed_2.shape

        torch.testing.assert_close(output_1, output_packed_1, atol=atol, rtol=rtol)
        torch.testing.assert_close(output_2, output_packed_2, atol=atol, rtol=rtol)


def test_end_to_end_packing(llama_config: ModelArgs):
    model = Transformer(llama_config).to("cuda")

    BS = 8
    SEQ_LEN = 128

    input_ = torch.randint(1, llama_config.vocab_size, (BS, SEQ_LEN)).to("cuda")

    seqlens = [torch.Tensor([SEQ_LEN // 4, SEQ_LEN // 4, SEQ_LEN // 2]).int().to("cuda") for _ in range(BS)]
    block_mask = create_block_mask_from_seqlens(seqlens)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_, block_mask=block_mask)

    assert output.shape == (BS, SEQ_LEN, llama_config.vocab_size)

    loss = output.mean()
    loss.backward()  # test that the backward for fa2


# --- Tests for HuggingFace Adapter vs Custom Model ---

@pytest.fixture
def hf_test_config_llama() -> Config:
    """Fixture for a minimal LLaMA config for HF adapter tests."""
    return Config(
        name_model="debugmodel",
        type_model="llama2",
        data=DataConfig(seq_length=128, fake=True),
        train=TrainConfig(reshard_after_forward=True, micro_bs=1),
        optim=OptimConfig(batch_size=1),
        hf_model_name=None, # Build from scratch
    )

@pytest.fixture
def hf_test_config_gpt2() -> Config:
    """Fixture for a minimal GPT-2 config for HF adapter tests."""
    return Config(
        name_model="debugmodel",
        type_model="gpt2",
        data=DataConfig(seq_length=128, fake=True),
        train=TrainConfig(reshard_after_forward=True, micro_bs=1),
        optim=OptimConfig(batch_size=1),
        hf_model_name=None, # Build from scratch
    )


def test_hf_llama_adapter_structure(hf_test_config_llama: Config):
    """Test if the HF LLaMA adapter exposes layers correctly for FSDP."""
    hf_model, hf_config = load_llama_model(hf_test_config_llama)

    assert hasattr(hf_model, "layers"), "Model should have a 'layers' attribute after adaptation."
    assert isinstance(hf_model.layers, dict), "'layers' attribute should be a dictionary."
    assert len(hf_model.layers) == hf_config.num_hidden_layers, "Number of layers in dict should match config."

    # Check that layers have the expected keys (string integers)
    for i in range(hf_config.num_hidden_layers):
        assert str(i) in hf_model.layers, f"Layer {i} not found in model.layers dictionary."
        assert isinstance(hf_model.layers[str(i)], torch.nn.Module), f"Layer {i} should be a torch Module."


def test_hf_gpt2_adapter_structure(hf_test_config_gpt2: Config):
    """Test if the HF GPT-2 adapter exposes layers correctly for FSDP."""
    hf_model, hf_config = load_gpt2_model(hf_test_config_gpt2)

    assert hasattr(hf_model, "layers"), "Model should have a 'layers' attribute after adaptation."
    assert isinstance(hf_model.layers, dict), "'layers' attribute should be a dictionary."
    # GPT2Config uses n_layer
    assert len(hf_model.layers) == hf_config.n_layer, "Number of layers in dict should match config."

    # Check that layers have the expected keys (string integers)
    for i in range(hf_config.n_layer):
        assert str(i) in hf_model.layers, f"Layer {i} not found in model.layers dictionary."
        assert isinstance(hf_model.layers[str(i)], torch.nn.Module), f"Layer {i} should be a torch Module."


@pytest.mark.parametrize("model_loader, config_fixture", [
    (load_llama_model, "hf_test_config_llama"),
    (load_gpt2_model, "hf_test_config_gpt2"),
])
def test_hf_adapter_forward_pass(model_loader, config_fixture, request):
    """Test the forward pass output shape and basic properties for HF adapters."""
    config: Config = request.getfixturevalue(config_fixture)
    hf_model, model_config = model_loader(config)
    hf_model.to("cuda")

    # Create test batch
    batch_size, seq_len = 2, config.data.seq_length
    # Use vocab_size from the loaded model config
    vocab_size = model_config.vocab_size
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")

    # Run forward pass
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = hf_model(tokens=tokens)

    # Check output shape and properties
    assert output.shape == (batch_size, seq_len, vocab_size), "Output shape mismatch."
    assert not torch.isnan(output).any(), "Output contains NaNs."
    assert not torch.isinf(output).any(), "Output contains Infs."


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 1, reason="Requires CUDA and at least 1 GPU")
@pytest.mark.parametrize("model_loader, config_fixture", [
    (load_llama_model, "hf_test_config_llama"),
    (load_gpt2_model, "hf_test_config_gpt2"),
])
def test_hf_adapter_fsdp_wrapping(model_loader, config_fixture, request):
    """Test if FSDP wrapping can be applied to the adapted HF model layers."""
    # Basic setup for FSDP (no distributed environment needed for this simple check)
    config: Config = request.getfixturevalue(config_fixture)
    hf_model, model_config = model_loader(config)
    hf_model.to("cuda") # Move model to CUDA before wrapping

    # Create a dummy device mesh and policy for testing wrapping logic
    # Note: This doesn't actually shard, just checks if the wrapping call succeeds structurally
    try:
        # Use a simplified mesh for local testing
        mesh = torch.distributed.device_mesh.init_device_mesh("cuda", (torch.cuda.device_count(),))
        local_mesh = mesh # In single-node, local mesh is the full mesh
    except Exception as e:
        pytest.skip(f"Skipping FSDP test due to device mesh init error: {e}")


    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)

    # Apply FSDP wrapping to transformer blocks
    num_layers = len(hf_model.layers)
    for layer_id_str, transformer_block in hf_model.layers.items():
        layer_id = int(layer_id_str)
        reshard_after_forward = config.train.reshard_after_forward and (layer_id < num_layers - 1)
        try:
            fully_shard(
                transformer_block,
                mp_policy=mp_policy,
                mesh=local_mesh,
                reshard_after_forward=reshard_after_forward,
            )
        except Exception as e:
            pytest.fail(f"FSDP wrapping failed for layer {layer_id}: {e}")

    # Apply FSDP wrapping to the whole model (embeddings, lm_head)
    try:
        fully_shard(
            hf_model,
            mp_policy=mp_policy,
            mesh=local_mesh,
            reshard_after_forward=config.train.reshard_after_forward,
        )
    except Exception as e:
        pytest.fail(f"FSDP wrapping failed for the main model: {e}")

    # Basic check: verify FSDP attributes are added (indicates wrapping occurred)
    # This check might be fragile depending on FSDP internal changes
    for layer in hf_model.layers.values():
         if not hasattr(layer, "_fsdp_params"):
             # Check if it's already compiled or has some other FSDP marker
             if not hasattr(layer, "__compiled_fn__") and not hasattr(layer, "_forward_hook_handle"):
                 pytest.fail(f"Layer {layer} does not seem to be FSDP wrapped.")

    if not hasattr(hf_model, "_fsdp_params"):
        if not hasattr(hf_model, "__compiled_fn__") and not hasattr(hf_model, "_forward_hook_handle"):
             pytest.fail("Main model does not seem to be FSDP wrapped.")


@pytest.mark.skip(reason="Conceptual test: Requires detailed setup and comparison logic")
def test_hf_adapter_training_step_equivalence():
    """
    Conceptual test: Compare loss and gradients between custom and HF adapter models.
    This requires careful setup:
    1. Initialize both models with the exact same weights.
    2. Use the same input batch.
    3. Use identical optimizer states.
    4. Run one training step.
    5. Compare resulting loss values (should be very close).
    6. Compare gradients for corresponding parameters (should be very close).
    """
    # 1. Setup Configs (ensure they produce compatible models)
    # 2. Load Custom Model (e.g., using get_model) with seed
    # 3. Load HF Adapter Model (e.g., using load_llama_model) with the same seed
    # 4. Ensure weights are identical (e.g., copy state dict or careful init)
    # 5. Create identical optimizers
    # 6. Create a sample batch
    # 7. Run forward/backward for custom model
    # 8. Run forward/backward for HF adapter model
    # 9. Assert loss values are close (torch.testing.assert_close)
    # 10. Iterate through parameters and assert gradients are close
    pass


@pytest.mark.skip(reason="Conceptual test: Performance/Memory tests are usually run manually or in dedicated benchmark suites")
def test_hf_adapter_performance_memory():
    """
    Conceptual test: Compare training throughput and peak memory usage.
    This is sensitive to hardware and environment, best run manually or in CI benchmarks.
    1. Setup models (custom and HF adapter).
    2. Run N training steps for each model, measuring time and peak GPU memory.
       (Use torch.cuda.reset_peak_memory_stats, torch.cuda.max_memory_allocated, time.time)
    3. Compare results, asserting that the HF adapter is within an acceptable threshold (e.g., <10% slower/higher memory).
    """
    pass
