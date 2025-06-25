# test_main.py

def test_main_output(capsys):
    from main import main
    main()
    captured = capsys.readouterr()
    assert "Hybrid Robot Navigation System: Hello, World!" in captured.out
