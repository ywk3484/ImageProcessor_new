import numpy as np

from scripts.read_speed import main


def test_read_speed_reads_the_whole_file_three_ways(tmp_path, capsys):
    import tifffile

    path = tmp_path / "s.tif"
    tifffile.imwrite(path, np.zeros((5, 16, 24), np.uint8), photometric="minisblack", metadata=None)
    assert main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "pages of 24 x 16 px, uint8, uncompressed" in out
    assert "(5 pages)" in out and out.count(" MB/s") == 4
    assert "cannot drop" not in out  # this system can make every read come from the disk
