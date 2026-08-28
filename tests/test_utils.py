import pytest
from stellar_sdk import Address, SorobanServer, xdr
from stellar_sdk.exceptions import ContractWasmRetrievalError, ExternalRefNotFoundError
from stellar_sdk.soroban_rpc import GetLedgerEntriesResponse, LedgerEntryResult

from stellar_contract_bindings import utils
from stellar_contract_bindings.metadata import get_token_sc_spec_entry

CONTRACT_ID = "CDOAW6D7NXAPOCO7TFAWZNJHK62E3IYRGNRVX3VOXNKNVOXCLLPJXQCF"
OWNER_ID = "CA3D5KRYM6CB7OWQ6TWYRR3Z4T7GNZLKERYNZGGA5SOAOPIFY6YQGAXE"
ACCOUNT_ID = "GD5KKP3LHUDXLDCGKP55NLEOEHMS3Z4BS6IDDZFCYU3BDXUZTBWL7JNF"
# A sentinel: every request the resolver makes is served by the mock, so no
# test can reach a real network even if the SDK's internal routing changes.
RPC_URL = "https://rpc.invalid"
WASM_HASH = bytes(range(32))


def _instance_entry(executable: xdr.ContractExecutable) -> LedgerEntryResult:
    """A contract instance ledger entry carrying the given executable."""
    instance = xdr.SCContractInstance(executable=executable, storage=None)
    data = xdr.LedgerEntryData(
        xdr.LedgerEntryType.CONTRACT_DATA,
        contract_data=xdr.ContractDataEntry(
            ext=xdr.ExtensionPoint(0),
            contract=Address(CONTRACT_ID).to_xdr_sc_address(),
            key=xdr.SCVal(xdr.SCValType.SCV_LEDGER_KEY_CONTRACT_INSTANCE),
            durability=xdr.ContractDataDurability.PERSISTENT,
            val=xdr.SCVal(xdr.SCValType.SCV_CONTRACT_INSTANCE, instance=instance),
        ),
    )
    return LedgerEntryResult(key="k", xdr=data.to_xdr(), lastModifiedLedgerSeq=1)


def _tag_entry(val: xdr.SCVal, tag: bytes = b"v1") -> LedgerEntryResult:
    """The owner contract's executable tag entry holding the given value."""
    data = xdr.LedgerEntryData(
        xdr.LedgerEntryType.CONTRACT_DATA,
        contract_data=xdr.ContractDataEntry(
            ext=xdr.ExtensionPoint(0),
            contract=Address(OWNER_ID).to_xdr_sc_address(),
            key=xdr.SCVal(
                xdr.SCValType.SCV_EXECUTABLE_TAG, executable_tag=xdr.SCString(tag)
            ),
            durability=xdr.ContractDataDurability.PERSISTENT,
            val=val,
        ),
    )
    return LedgerEntryResult(key="k", xdr=data.to_xdr(), lastModifiedLedgerSeq=1)


def _external_ref(owner: str = OWNER_ID, tag: bytes = b"v1") -> xdr.ContractExecutable:
    return xdr.ContractExecutable(
        xdr.ContractExecutableType.CONTRACT_EXECUTABLE_EXTERNAL_REF,
        external_ref=xdr.ContractExecutableExternalRef(
            executable_owner=Address(owner).to_xdr_sc_address(),
            tag=xdr.SCString(tag),
        ),
    )


def _wasm_hash_val(wasm_hash: bytes = WASM_HASH) -> xdr.SCVal:
    return xdr.SCVal(xdr.SCValType.SCV_BYTES, bytes=xdr.SCBytes(wasm_hash))


def _serve_entries(monkeypatch, responses):
    """Answer successive get_ledger_entries calls from the given lists.

    Everything the resolver requests, including the SDK's tag lookup, goes
    through get_ledger_entries, so this one seam intercepts every request.
    Returns the list of key lists received, for asserting on the requests.
    """
    calls = []

    def fake_get_ledger_entries(self, keys):
        calls.append(keys)
        return GetLedgerEntriesResponse(
            entries=responses[len(calls) - 1], latestLedger=1
        )

    monkeypatch.setattr(SorobanServer, "get_ledger_entries", fake_get_ledger_entries)
    return calls


class TestGetSpecsByContractId:
    def test_external_ref_resolves_to_wasm_specs(self, monkeypatch):
        calls = _serve_entries(
            monkeypatch,
            [[_instance_entry(_external_ref())], [_tag_entry(_wasm_hash_val())]],
        )
        received = []

        def fake_get_specs_by_wasm_hash(wasm_hash, rpc_url):
            received.append((wasm_hash, rpc_url))
            return ["spec"]

        # Patched at module level so the code read stays out of the ledger mock.
        monkeypatch.setattr(
            utils, "get_specs_by_wasm_hash", fake_get_specs_by_wasm_hash
        )
        assert utils.get_specs_by_contract_id(CONTRACT_ID, RPC_URL) == ["spec"]
        assert received == [(WASM_HASH, RPC_URL)]
        assert len(calls) == 2

    def test_binary_tag_reaches_the_ledger_key_verbatim(self, monkeypatch):
        tag = b"\xff\xfe\x00"
        calls = _serve_entries(
            monkeypatch,
            [
                [_instance_entry(_external_ref(tag=tag))],
                [_tag_entry(_wasm_hash_val(), tag=tag)],
            ],
        )
        monkeypatch.setattr(
            utils, "get_specs_by_wasm_hash", lambda wasm_hash, rpc_url: ["spec"]
        )
        utils.get_specs_by_contract_id(CONTRACT_ID, RPC_URL)
        tag_key = calls[1][0].contract_data
        assert tag_key.key.executable_tag.sc_string == tag
        assert tag_key.durability == xdr.ContractDataDurability.PERSISTENT
        assert Address.from_xdr_sc_address(tag_key.contract).address == OWNER_ID

    def test_non_contract_owner_raises_without_a_tag_request(self, monkeypatch):
        calls = _serve_entries(
            monkeypatch, [[_instance_entry(_external_ref(owner=ACCOUNT_ID))]]
        )
        with pytest.raises(ValueError, match="is not a contract"):
            utils.get_specs_by_contract_id(CONTRACT_ID, RPC_URL)
        assert len(calls) == 1

    def test_missing_tag_entry_raises(self, monkeypatch):
        _serve_entries(monkeypatch, [[_instance_entry(_external_ref())], []])
        with pytest.raises(ExternalRefNotFoundError):
            utils.get_specs_by_contract_id(CONTRACT_ID, RPC_URL)

    @pytest.mark.parametrize(
        "val",
        [
            xdr.SCVal(xdr.SCValType.SCV_U32, u32=xdr.Uint32(1)),
            _wasm_hash_val(b"\x01" * 31),
            _wasm_hash_val(b"\x00" * 32),
        ],
        ids=["wrong-arm", "wrong-length", "all-zero"],
    )
    def test_malformed_tag_value_raises(self, monkeypatch, val):
        _serve_entries(
            monkeypatch, [[_instance_entry(_external_ref())], [_tag_entry(val)]]
        )
        with pytest.raises(ContractWasmRetrievalError) as exc:
            utils.get_specs_by_contract_id(CONTRACT_ID, RPC_URL)
        # ExternalRefNotFoundError subclasses ContractWasmRetrievalError, so
        # the exact class is asserted to keep this case distinct from a
        # missing entry.
        assert type(exc.value) is ContractWasmRetrievalError

    def test_wasm_instance_resolves_as_before(self, monkeypatch):
        executable = xdr.ContractExecutable(
            xdr.ContractExecutableType.CONTRACT_EXECUTABLE_WASM,
            wasm_hash=xdr.Hash(WASM_HASH),
        )
        calls = _serve_entries(monkeypatch, [[_instance_entry(executable)]])
        received = []

        def fake_get_specs_by_wasm_hash(wasm_hash, rpc_url):
            received.append(wasm_hash)
            return ["spec"]

        monkeypatch.setattr(
            utils, "get_specs_by_wasm_hash", fake_get_specs_by_wasm_hash
        )
        assert utils.get_specs_by_contract_id(CONTRACT_ID, RPC_URL) == ["spec"]
        assert received == [WASM_HASH]
        assert len(calls) == 1

    def test_sac_instance_returns_the_embedded_token_spec(self, monkeypatch):
        executable = xdr.ContractExecutable(
            xdr.ContractExecutableType.CONTRACT_EXECUTABLE_STELLAR_ASSET
        )
        _serve_entries(monkeypatch, [[_instance_entry(executable)]])
        specs = utils.get_specs_by_contract_id(CONTRACT_ID, RPC_URL)
        assert [s.to_xdr() for s in specs] == [
            s.to_xdr() for s in get_token_sc_spec_entry()
        ]

    def test_contract_not_found_raises(self, monkeypatch):
        _serve_entries(monkeypatch, [[]])
        with pytest.raises(ValueError, match="Contract not found"):
            utils.get_specs_by_contract_id(CONTRACT_ID, RPC_URL)


def test_get_specs_by_wasm_hash_raises_when_wasm_not_found(monkeypatch):
    _serve_entries(monkeypatch, [[]])
    with pytest.raises(ValueError, match="Wasm not found"):
        utils.get_specs_by_wasm_hash(WASM_HASH, RPC_URL)
