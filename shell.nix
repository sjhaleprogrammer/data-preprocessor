{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = [
    (pkgs.python311.withPackages (ps: with ps; [
      numpy
      pandas
      beautifulsoup4
      pyarrow
      cython
      psutil
      openpyxl
      setuptools
      # Add other Python packages here
    ]))
  ];
}