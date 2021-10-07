contract Mint {
    mapping(uint256 => uint256) _balances;
    uint256 _totalSupply;

    function _mint(uint256 account, uint256 value) public {
        require(account != 0);
        require(_totalSupply + value >= _totalSupply);
        _totalSupply = _totalSupply + value;
        // should be safe since sum(_balances) = _totalSupply
        _balances[account] = _balances[account] + value;
    }
}
